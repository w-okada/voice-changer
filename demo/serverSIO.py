import eventlet
import socketio
import sys, os, struct, argparse, logging
from distutils.util import strtobool
from datetime import datetime
from OpenSSL import SSL, crypto

import torch
import numpy as np
from scipy.io.wavfile import write

sys.path.append("/MMVC_Trainer")
sys.path.append("/MMVC_Trainer/text")


import utils
import commons 
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from mel_processing import spectrogram_torch
from text import text_to_sequence, cleaned_text_to_sequence

class MyCustomNamespace(socketio.Namespace): 
    def __init__(self, namespace, config, model):
        super().__init__(namespace)
        self.hps =utils.get_hparams_from_file(config)
        self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model)
        self.net_g.eval()
        self.gpu_num = torch.cuda.device_count()
        print("GPU_NUM:",self.gpu_num)
        utils.load_checkpoint( model, self.net_g, None)

    def on_connect(self, sid, environ):
        print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))
        # print('[{}] connet env : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , environ))

    def on_request_message(self, sid, msg): 
        # print("MESSGaa", msg)
        gpu = int(msg[0])
        srcId = int(msg[1])
        dstId = int(msg[2])
        timestamp = int(msg[3])
        data = msg[4]
        # print(srcId, dstId, timestamp)
        unpackedData = np.array(struct.unpack('<%sh'%(len(data) // struct.calcsize('<h') ), data))
        write("logs/received_data.wav", 24000, unpackedData.astype(np.int16))

        # self.emit('response', msg)

        if gpu<0 or self.gpu_num==0 :
            with torch.no_grad():
                
                text_norm = text_to_sequence("a", self.hps.data.text_cleaners)
                text_norm = commons.intersperse(text_norm, 0)
                text_norm = torch.LongTensor(text_norm)

                audio = torch.FloatTensor(unpackedData.astype(np.float32))
                audio_norm = audio /self.hps.data.max_wav_value
                audio_norm = audio_norm.unsqueeze(0)


                spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
                        self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                        center=False)
                spec = torch.squeeze(spec, 0)
                sid = torch.LongTensor([int(srcId)])
                
                data =  (text_norm, spec, audio_norm, sid)

                data = TextAudioSpeakerCollate()([data])
                x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cpu() for x in data]
                sid_tgt1 = torch.LongTensor([dstId]).cpu()
                audio1 = (self.net_g.cpu().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data * self.hps.data.max_wav_value).cpu().float().numpy()
        else:
            with torch.no_grad():

                text_norm = text_to_sequence("a", self.hps.data.text_cleaners)
                text_norm = commons.intersperse(text_norm, 0)
                text_norm = torch.LongTensor(text_norm)

                audio = torch.FloatTensor(unpackedData.astype(np.float32))
                audio_norm = audio /self.hps.data.max_wav_value
                audio_norm = audio_norm.unsqueeze(0)


                spec = spectrogram_torch(audio_norm, self.hps.data.filter_length,
                        self.hps.data.sampling_rate, self.hps.data.hop_length, self.hps.data.win_length,
                        center=False)
                spec = torch.squeeze(spec, 0)
                sid = torch.LongTensor([int(srcId)])
                
                data =  (text_norm, spec, audio_norm, sid)
                data = TextAudioSpeakerCollate()([data])

                x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda(gpu) for x in data]
                sid_tgt1 = torch.LongTensor([dstId]).cuda(gpu)
                audio1 = (self.net_g.cuda(gpu).voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data * self.hps.data.max_wav_value).cpu().float().numpy()

        audio1 = audio1.astype(np.int16)
        bin = struct.pack('<%sh'%len(audio1), *audio1)

        # print("return timestamp", timestamp)        
        self.emit('response',[timestamp, bin])




    def on_disconnect(self, sid):
        # print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        pass;

def setupArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=8080, help="port")
    parser.add_argument("-c", type=str, required=True, help="path for the config.json")
    parser.add_argument("-m", type=str, required=True, help="path for the model file")
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
    CONFIG = args.c
    MODEL  = args.m

    printMessage(f"Start MMVC SocketIO Server", level=0)
    printMessage(f"CONFIG:{CONFIG}, MODEL:{MODEL}", level=1)

    if os.environ["EX_PORT"]:
        EX_PORT = os.environ["EX_PORT"]
        printMessage(f"External_Port:{EX_PORT} Internal_Port:{PORT}", level=1)
    else:
        printMessage(f"Internal_Port:{PORT}", level=1)

    if os.environ["EX_IP"]:
        EX_IP = os.environ["EX_IP"]
        printMessage(f"External_IP:{EX_IP}", level=1)

    if args.https and args.httpsSelfSigned == 1:
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
    elif args.https and args.httpsSelfSigned == 0:
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
    sio.register_namespace(MyCustomNamespace('/test', CONFIG, MODEL)) 
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

