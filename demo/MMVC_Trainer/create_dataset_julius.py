import glob
import sys

def read_lab(lab_f):
    with open(lab_f, 'r') as f:
        kw_list = f.read().split("\n")

    out_phono = []
    for i in range(len(kw_list)-1):
        out_phono.append(kw_list[i].split()[2])
        out_phono.append("-")

    if out_phono[0] == 'silB' and out_phono[-2] == 'silE':
        out_phono[0] = 'sil'
        out_phono[-2] = 'sil'
        out_phono = out_phono[0:-1]
        out_phono_str = "".join(out_phono)
        return out_phono_str

    else:
        print("Error!")
        exit

def create_dataset(filename):
    speaker_id = 0
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
        wav_file_list = glob.glob(d+"/wav/*")
        lab_file_list = glob.glob(d + "/text/*")
        wav_file_list.sort()
        lab_file_list.sort()
        if len(wav_file_list) == 0:
            continue
        counter = 0
        for lab, wav in zip(lab_file_list, wav_file_list):
            test = read_lab(lab)
            print(wav + "|"+ str(speaker_id) + "|"+ test)
            if counter % 10 != 0:
                output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
            else:
                output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+d + "\n")
        speaker_id = speaker_id + 1

    for d in textless_dir_list:
        wav_file_list = glob.glob(d+"/*")
        wav_file_list.sort()
        counter = 0
        for wav in wav_file_list:
            print(wav + "|"+ str(speaker_id) + "|a")
            if counter % 10 != 0:
                output_file_list_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            else:
                output_file_list_val_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+d + "\n")
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
    return speaker_id -1

def main(argv):
    filename = str(sys.argv[1])
    print(filename)
    n_spk = create_dataset(filename)
    return filename, n_spk

if __name__ == '__main__':
    sys.exit(main(sys.argv))