import glob
import sys
import os
import argparse
import pyopenjtalk
import json

def mozi2phone(mozi):
    text = pyopenjtalk.g2p(mozi)
    text = "sil " + text + " sil"
    text = text.replace(' ', '-')
    return text

def create_json(filename, num_speakers, sr, config_path):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data['data']['training_files'] = 'filelists/' + filename + '_textful.txt'
    data['data']['validation_files'] = 'filelists/' + filename + '_textful_val.txt'
    data['data']['training_files_notext'] = 'filelists/' + filename + '_textless.txt'
    data['data']['validation_files_notext'] = 'filelists/' + filename + '_val_textless.txt'
    data['data']['sampling_rate'] = sr
    data['data']['n_speakers'] = num_speakers

    with open("./configs/" + filename + ".json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_dataset(filename, my_sid):
    speaker_id = my_sid
    textful_dir_list = glob.glob("dataset/textful/*")
    textless_dir_list = glob.glob("dataset/textless/*")
    textful_dir_list.sort()
    textless_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    output_file_list_textless = list()
    output_file_list_val_textless = list()
    for d in textful_dir_list:
        wav_file_list = glob.glob(d+"/wav/*.wav")
        lab_file_list = glob.glob(d + "/text/*.txt")
        wav_file_list.sort()
        lab_file_list.sort()
        if len(wav_file_list) == 0:
            continue
        counter = 0
        for lab, wav in zip(lab_file_list, wav_file_list):
            with open(lab, 'r', encoding="utf-8") as f:
                mozi = f.read().split("\n")
            print(str(mozi))
            test = mozi2phone(str(mozi))
            print(test)
            print(wav + "|"+ str(speaker_id) + "|"+ test)
            if counter % 10 != 0:
                output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
            else:
                output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1
        if speaker_id > 108:
            break

    for d in textless_dir_list:
        wav_file_list = glob.glob(d+"/*.wav")
        wav_file_list.sort()
        counter = 0
        for wav in wav_file_list:
            print(wav + "|"+ str(speaker_id) + "|a")
            if counter % 10 != 0:
                output_file_list_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            else:
                output_file_list_val_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1

    with open('filelists/' + filename + '_textful.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list)
    with open('filelists/' + filename + '_textful_val.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val)
    with open('filelists/' + filename + '_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_textless)
    with open('filelists/' + filename + '_val_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val_textless)
    with open('filelists/' + filename + '_Correspondence.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(Correspondence_list)
    return speaker_id + 1

def create_dataset(filename, my_sid):
    textful_dir_list = glob.glob("dataset/textful/*")
    textless_dir_list = glob.glob("dataset/textless/*")
    textful_dir_list.sort()
    textless_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    output_file_list_textless = list()
    output_file_list_val_textless = list()
    #paths
    my_path = "dataset/textful/00_myvoice"
    target_path = "dataset/textful/01_target"
    print("myvoice : {}".format(str(os.path.isdir(my_path))))
    print("target_path : {}".format(str(os.path.isdir(target_path))))

    #set list wav and text
    #myvoice
    speaker_id = my_sid
    d = my_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    lab_file_list = glob.glob(d + "/text/*.txt")
    wav_file_list.sort()
    lab_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    if len(lab_file_list) == 0:
        print("Error : " + d + "/text にテキストデータがありません")
        exit()
    counter = 0

    for lab, wav in zip(lab_file_list, wav_file_list):
        with open(lab, 'r', encoding="utf-8") as f:
            mozi = f.read().split("\n")
        print(str(mozi))
        test = mozi2phone(str(mozi))
        print(test)
        print(wav + "|"+ str(speaker_id) + "|"+ test)
        if counter % 10 != 0:
            output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
        else:
            output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    speaker_id = 108
    d = target_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    lab_file_list = glob.glob(d + "/text/*.txt")
    wav_file_list.sort()
    lab_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for lab, wav in zip(lab_file_list, wav_file_list):
      with open(lab, 'r', encoding="utf-8") as f:
          mozi = f.read().split("\n")
      print(str(mozi))
      test = mozi2phone(str(mozi))
      print(test)
      print(wav + "|"+ str(speaker_id) + "|"+ test)
      if counter % 10 != 0:
          output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
      else:
          output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
      counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    for d in textless_dir_list:
        wav_file_list = glob.glob(d+"/*.wav")
        wav_file_list.sort()
        counter = 0
        for wav in wav_file_list:
            print(wav + "|"+ str(speaker_id) + "|a")
            if counter % 10 != 0:
                output_file_list_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            else:
                output_file_list_val_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1

    with open('filelists/' + filename + '_textful.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list)
    with open('filelists/' + filename + '_textful_val.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val)
    with open('filelists/' + filename + '_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_textless)
    with open('filelists/' + filename + '_val_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val_textless)
    with open('filelists/' + filename + '_Correspondence.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(Correspondence_list)
    return 110

def create_dataset_zundamon(filename, my_sid):
    textful_dir_list = glob.glob("dataset/textful/*")
    textless_dir_list = glob.glob("dataset/textless/*")
    textful_dir_list.sort()
    textless_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    output_file_list_textless = list()
    output_file_list_val_textless = list()
    #paths
    my_path = "dataset/textful/00_myvoice"
    zundamon_path = "dataset/textful/1205_zundamon"
    print("myvoice : {}".format(str(os.path.isdir(my_path))))
    print("zundamon_path : {}".format(str(os.path.isdir(zundamon_path))))

    #set list wav and text
    #myvoice
    speaker_id = my_sid
    d = my_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    lab_file_list = glob.glob(d + "/text/*.txt")
    wav_file_list.sort()
    lab_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    if len(lab_file_list) == 0:
        print("Error : " + d + "/text にテキストデータがありません")
        exit()
    counter = 0

    for lab, wav in zip(lab_file_list, wav_file_list):
        with open(lab, 'r', encoding="utf-8") as f:
            mozi = f.read().split("\n")
        print(str(mozi))
        test = mozi2phone(str(mozi))
        print(test)
        print(wav + "|"+ str(speaker_id) + "|"+ test)
        if counter % 10 != 0:
            output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
        else:
            output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    speaker_id = 100
    d = zundamon_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    lab_file_list = glob.glob(d + "/text/*.txt")
    wav_file_list.sort()
    lab_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for lab, wav in zip(lab_file_list, wav_file_list):
      with open(lab, 'r', encoding="utf-8") as f:
          mozi = f.read().split("\n")
      print(str(mozi))
      test = mozi2phone(str(mozi))
      print(test)
      print(wav + "|"+ str(speaker_id) + "|"+ test)
      if counter % 10 != 0:
          output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
      else:
          output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
      counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    for d in textless_dir_list:
        wav_file_list = glob.glob(d+"/*.wav")
        wav_file_list.sort()
        counter = 0
        for wav in wav_file_list:
            print(wav + "|"+ str(speaker_id) + "|a")
            if counter % 10 != 0:
                output_file_list_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            else:
                output_file_list_val_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1

    with open('filelists/' + filename + '_textful.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list)
    with open('filelists/' + filename + '_textful_val.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val)
    with open('filelists/' + filename + '_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_textless)
    with open('filelists/' + filename + '_val_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val_textless)
    with open('filelists/' + filename + '_Correspondence.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(Correspondence_list)
    return 110

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='filelist for configuration')
    parser.add_argument('-s', '--sr', type=int, default=24000,
                        help='sampling rate (default = 24000)')
    parser.add_argument('-m', '--mysid', type=int, default=107,
                        help='sampling rate (default = 24000)')
    parser.add_argument('-z', '--zundamon', type=bool, default=False,
                        help='U.N. zundamon Was Her? (default = False)')
    parser.add_argument('-c', '--config', type=str, default="./configs/baseconfig.json",
                        help='JSON file for configuration')
    args = parser.parse_args()
    filename = args.filename
    print(filename)
    if args.zundamon:
        n_spk = create_dataset_zundamon(filename, args.mysid)
    else:
        n_spk = create_dataset(filename, args.mysid)
    
    create_json(filename, n_spk, args.sr, args.config)

if __name__ == '__main__':
    main()
