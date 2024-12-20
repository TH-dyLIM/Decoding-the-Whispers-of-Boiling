""" 코드 edition 요약 """
# 5th edition : 24.08.05 YDY
# 1st edition): Thermosyphon AE Hit voltage data 병합 후, AE Waveform 생성 및 Spectrogram 변환 코드
# 3rd edition): 유동비등 AE Hit voltage data 병합 후, waveform, spectrogram, wavelet 변환 코드
# 4th edition): CNN 학습을 위한 spectrogram 변환 코드. 기존 전체 waveform을 사용하는 대신 0.1초의 waveform만 변환함.
# 현재코드(5th edition): CNN 학습을 위한 spectrogram 변환 코드. 병합된 전체 waveform을 spec_cnn으로 변환함.

# AE 측정 조건: sampling rate - 1,000 kPs, Hit당 샘플링 수 - 10,240, Hit duration - 10.480 ms
# AE 센서: R15a
# nfft = 1000
# win = 50
# hop = 50
# frmax = 300000
# dbmin = 30
# dbmax = 90

###########################################################################################################
"""
1460개 변환에 약 11분 걸림.
1개당 0.45초 수준
"""
###########################################################################################################
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# spectrogram 저장 함수
def spectrogram_for_cnn(Spectrogram_dB, Tot_Time, path, sr, hop, dbmin, dbmax, frmax):
    try:
        # print(f"Generating spectrogram for {path}")  # 디버깅 메시지 추가
        S_dB = Spectrogram_dB
        # print(f"Spectrogram shape: {S_dB.shape}")  # 디버깅 메시지 추가
        plt.figure(figsize=(2.56, 2.56))
        librosa.display.specshow(S_dB, sr=sr, hop_length=hop, y_axis='linear', x_axis='s')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        plt.clim(dbmin, dbmax)
        plt.jet()
        plt.xlim(0, Tot_Time)
        plt.ylim(0, frmax)
        plt.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved spectrogram to {path}")  # 디버깅 메시지 추가
    except Exception as e:
        print(f"Error generating spectrogram for {path}: {e}")
        print(traceback.format_exc())  # 상세한 오류 메시지 출력

# 데이터셋 클래스 정의
class AEDataset(Dataset):
    def __init__(self, csv_files, sample_rate, nfft, win, hop, frmax, dbmin, dbmax):
        self.csv_files = csv_files
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.win = win
        self.hop = hop
        self.frmax = frmax
        self.dbmin = dbmin
        self.dbmax = dbmax

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        try:
            csv_file = self.csv_files[idx]
            # print(f"Reading {csv_file}")  # 디버깅 메시지 추가
            data_frame = pd.read_csv(csv_file)

            if 'Voltage' in data_frame.columns:
                data = np.array(data_frame['Voltage'].values, dtype='float32')
            else:
                data = np.array(data_frame.iloc[:, 1].values, dtype='float32')

            # GPU로 데이터 전송
            data = torch.tensor(data).cuda()

            # STFT 수행
            window = torch.hann_window(self.win).cuda()
            AE_stft = torch.stft(data, n_fft=self.nfft, win_length=self.win, hop_length=self.hop, window=window, return_complex=True)
            AE_stft_abs = torch.abs(AE_stft)
            Spectrogram_dB = librosa.power_to_db(AE_stft_abs.cpu().numpy(), ref=0.000001)

            Tot_Time = len(data) / self.sample_rate

            # print(f"Processed {csv_file}, Spectrogram shape: {Spectrogram_dB.shape}")  # 디버깅 메시지 추가

            return Spectrogram_dB, Tot_Time, csv_file
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            print(traceback.format_exc())  # 상세한 오류 메시지 출력
            return None

    def process_data(self, index):
        return self[index]

def save_spectrogram(data):
    try:
        if data is None:
            return
        Spectrogram_dB, Tot_Time, csv_file = data
        file_name = os.path.basename(csv_file)
        output_path = os.path.join(output_subfolder, os.path.splitext(file_name)[0] + '.png')
        print(f"Attempting to save spectrogram for {file_name} to {output_path}")

        # 파일 저장 전 디렉토리 존재 확인
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        spectrogram_for_cnn(Spectrogram_dB, Tot_Time, output_path, sample_rate, hop, dbmin, dbmax, frmax)

        # 파일이 실제로 저장되었는지 확인
        if os.path.exists(output_path):
            print(f"Successfully saved spectrogram to {output_path}")
        else:
            print(f"Failed to save spectrogram to {output_path}")
    except Exception as e:
        print(f"Error saving spectrogram for {csv_file if 'csv_file' in locals() else 'unknown file'}: {str(e)}")
        print(traceback.format_exc())  # 상세한 오류 메시지 출력

def main():
    try:
        # 주요 변수 설정
        global sample_rate, hop, dbmin, dbmax, frmax, output_subfolder
        sample_rate = 1000000  # Sampling rate 1000 kHz
        nfft = 1000
        win = 50
        hop = 50
        frmax = 300000
        dbmin = 30
        dbmax = 90

        # 입력 및 출력 폴더 설정
        input_folder = './dataset_signal_pool_train'
        output_folder = './spectrograms'

        # 입력 폴더 이름 추출
        input_folder_name = os.path.basename(input_folder)

        # 출력 폴더 생성 (spectrograms/input_folder_name)
        output_subfolder = os.path.join(output_folder, input_folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # CSV 파일 목록 가져오기
        csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
        print(f"Found {len(csv_files)} CSV files")  # 디버그 메시지 추가

        # 데이터셋 및 데이터로더 생성
        dataset = AEDataset(csv_files, sample_rate, nfft, win, hop, frmax, dbmin, dbmax)
        print(f"Dataset length: {len(dataset)}")  # 디버그 메시지 추가

        # 병렬로 데이터를 처리하고 메인 스레드에서 spectrogram 저장
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(dataset.process_data, idx) for idx in range(len(dataset))]
            for future in as_completed(futures):
                try:
                    result = future.result()  # 결과를 기다림으로써 예외 발생 시 확인
                    save_spectrogram(result)
                except Exception as e:
                    print(f"Error in future: {e}")
                    print(traceback.format_exc())  # 상세한 오류 메시지 출력

        print("모든 CSV 파일의 spectrogram 변환이 완료되었습니다.")
    except Exception as e:
        print(f"Error in main function: {e}")
        print(traceback.format_exc())  # 상세한 오류 메시지 출력

if __name__ == '__main__':
    main()

