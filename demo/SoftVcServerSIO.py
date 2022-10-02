import eventlet
import socketio
import sys,os , math, struct, argparse, logging
from distutils.util import strtobool
from datetime import datetime
from OpenSSL import SSL, crypto

import torch, torchaudio
import numpy as np
from scipy.io.wavfile import write, read


sys.path.append("/hubert")
from hubert import hubert_discrete, hubert_soft, kmeans100

sys.path.append("/acoustic-model")
from acoustic import hubert_discrete, hubert_soft

sys.path.append("/hifigan")
from hifigan import hifigan

hubert_model = torch.load("/models/bshall_hubert_main.pt").cuda()
acoustic_model = torch.load("/models/bshall_acoustic-model_main.pt").cuda()
hifigan_model = torch.load("/models/bshall_hifigan_main.pt").cuda()


def applyVol(i, chunk, vols):
  curVol = vols[i] / 2
  if curVol < 0.0001:
    line = torch.zeros(chunk.size())
  else:
    line = torch.ones(chunk.size())

  volApplied = torch.mul(line, chunk)  
  volApplied = volApplied.unsqueeze(0)
  return volApplied


class MyCustomNamespace(socketio.Namespace): 
    def __init__(self, namespace):
        super().__init__(namespace)

    def on_connect(self, sid, environ):
        print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))

    def on_request_message(self, sid, msg): 
        # print("Processing Request...")
        gpu = int(msg[0])
        srcId = int(msg[1])
        dstId = int(msg[2])
        timestamp = int(msg[3])
        data = msg[4]
        # print(srcId, dstId, timestamp)
        unpackedData = np.array(struct.unpack('<%sh'%(len(data) // struct.calcsize('<h') ), data))
        write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

        source, sr = torchaudio.load("logs/received_data.wav") # デフォルトでnormalize=Trueがついており、float32に変換して読んでくれるらしいのでこれを使う。https://pytorch.org/audio/stable/backend.html

        source_16k = torchaudio.functional.resample(source, 24000, 16000)
        source_16k = source_16k.unsqueeze(0).cuda()
        # SOFT-VC
        with torch.inference_mode():
            units = hubert_model.units(source_16k)
            mel = acoustic_model.generate(units).transpose(1, 2)
            target = hifigan_model(mel)

        dest = torchaudio.functional.resample(target, 16000,24000)
        dest = dest.squeeze().cpu()

        # ソースの音量取得
        source = source.cpu()
        specgram = torchaudio.transforms.MelSpectrogram(sample_rate=24000)(source)
        vol_apply_window_size = math.ceil(len(source[0]) / specgram.size()[2])
        specgram = specgram.transpose(1,2)
        vols = [ torch.max(i) for i in specgram[0]]
        chunks = torch.split(dest, vol_apply_window_size,0)

        chunks = [applyVol(i,c,vols) for i, c in enumerate(chunks)]
        dest = torch.cat(chunks,1)
        arr = np.array(dest.squeeze())

        int_size = 2**(16 - 1) - 1
        arr = (arr * int_size).astype(np.int16)
        bin = struct.pack('<%sh'%len(arr), *arr)

        self.emit('response',[timestamp, bin])

    def on_disconnect(self, sid):
        pass;

def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8080, help="port")
    parser.add_argument("--https", type=strtobool, default=False, help="use https")
    parser.add_argument("--httpsKey", type=str, default="ssl.key", help="path for the key of https")
    parser.add_argument("--httpsCert", type=str, default="ssl.cert", help="path for the cert of https")
    parser.add_argument("--httpsSelfSigned", type=strtobool, default=True, help="generate self-signed certificate")
    return parser

def create_self_signed_cert(certfile, keyfile, certargs, cert_dir="."):
    C_F = os.path.join(cert_dir, certfile)
    K_F = os.path.join(cert_dir, keyfile)
    if not os.path.exists(C_F) or not os.path.exists(K_F):
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        cert = crypto.X509()
        cert.get_subject().C = certargs["Country"]
        cert.get_subject().ST = certargs["State"]
        cert.get_subject().L = certargs["City"]
        cert.get_subject().O = certargs["Organization"]
        cert.get_subject().OU = certargs["Org. Unit"]
        cert.get_subject().CN = 'Example'
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(315360000)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha1')
        open(C_F, "wb").write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        open(K_F, "wb").write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))


def printMessage(message, level=0):
    if level == 0:
        print(f"\033[17m{message}\033[0m")
    elif level == 1:
        print(f"\033[34m    {message}\033[0m")
    elif level == 2:
        print(f"\033[32m    {message}\033[0m")
    else:
        print(f"\033[47m    {message}\033[0m")

if __name__ == '__main__':
    parser = setupArgParser()
    args = parser.parse_args()
    PORT = args.p

    printMessage(f"Start SoftVC SocketIO Server", level=0)

    if os.environ["EX_PORT"]:
        EX_PORT = os.environ["EX_PORT"]
        printMessage(f"External_Port:{EX_PORT} Internal_Port:{PORT}", level=1)
    else:
        printMessage(f"Internal_Port:{PORT}", level=1)

    if os.environ["EX_IP"]:
        EX_IP = os.environ["EX_IP"]
        printMessage(f"External_IP:{EX_IP}", level=1)

    if args.https == 1 and args.httpsSelfSigned == 1:
        # HTTPS(おれおれ証明書生成) 
        os.makedirs("./key", exist_ok=True)
        key_base_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        keyname = f"{key_base_name}.key"
        certname = f"{key_base_name}.cert"
        create_self_signed_cert(certname, keyname, certargs=
                            {"Country": "JP",
                             "State": "Tokyo",
                             "City": "Chuo-ku",
                             "Organization": "F",
                             "Org. Unit": "F"}, cert_dir="./key")
        key_path = os.path.join("./key", keyname)
        cert_path = os.path.join("./key", certname)
        printMessage(f"protocol: HTTPS(self-signed), key:{key_path}, cert:{cert_path}", level=1)

    elif args.https == 1 and args.httpsSelfSigned == 0:
        # HTTPS 
        key_path = args.httpsKey
        cert_path = args.httpsCert
        printMessage(f"protocol: HTTPS, key:{key_path}, cert:{cert_path}", level=1)
    else:
        # HTTP
        printMessage(f"protocol: HTTP", level=1)


    # アドレス表示
    if args.https == 1:
        printMessage(f"open https://<IP>:<PORT>/ with your browser.", level=0)
    else:
        printMessage(f"open http://<IP>:<PORT>/ with your browser.", level=0)

    if EX_PORT and EX_IP and args.https == 1:
        printMessage(f"In many cases it is one of the following", level=1)
        printMessage(f"https://localhost:{EX_PORT}/", level=1)
        for ip in EX_IP.strip().split(" "):
            printMessage(f"https://{ip}:{EX_PORT}/", level=1)
    elif EX_PORT and EX_IP and args.https == 0:
        printMessage(f"In many cases it is one of the following", level=1)
        printMessage(f"http://localhost:{EX_PORT}/", level=1)
        # for ip in EX_IP.strip().split(" "):
        #     print(f"    http://{ip}:{EX_PORT}/")

    # SocketIOセットアップ
    sio = socketio.Server(cors_allowed_origins='*') 
    sio.register_namespace(MyCustomNamespace('/test')) 
    app = socketio.WSGIApp(sio,static_files={
        '': '../frontend/dist',
        '/': '../frontend/dist/index.html',
    }) 

    ### log を設定すると通常出力されないログが取得できるようだ。（ログ出力抑制には役立たない?）
    # logger = logging.getLogger("logger")
    # logger.propagate=False
    # handler = logging.FileHandler(filename="logger.log")
    # logger.addHandler(handler)


    if args.https:
        # HTTPS サーバ起動
        sslWrapper = eventlet.wrap_ssl(
                eventlet.listen(('0.0.0.0',int(PORT))),
                certfile=cert_path, 
                keyfile=key_path, 
                # server_side=True
            )
        ### log を設定すると通常出力されないログが取得できるようだ。（ログ出力抑制には役立たない?）
        # eventlet.wsgi.server(sslWrapper, app, log=logger)     
        eventlet.wsgi.server(sslWrapper, app)     
    else:
        # HTTP サーバ起動
        ### log を設定すると通常出力されないログが取得できるようだ。（ログ出力抑制には役立たない?）
        # eventlet.wsgi.server(eventlet.listen(('0.0.0.0',int(PORT))), app, log=logger) 
        eventlet.wsgi.server(eventlet.listen(('0.0.0.0',int(PORT))), app) 



