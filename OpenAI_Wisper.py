#pip install git+https://github.com/openai/whisper.git 
import argparse
import whisper
import auditok
import time

class Whisper:
    def __init__(self,model_name='medium',device_mode='cuda'): #'cpu' ,'cuda'
        self.model_name=model_name
        if self.model_name =='small':
            self.model = whisper.load_model('./small.pt', device=device_mode)
        if self.model_name == 'base':
            self.model = whisper.load_model('./base.pt', device=device_mode)
        if self.model_name == 'medium':
            self.model = whisper.load_model('./medium.pt', device=device_mode)

    def run_whisper(self,audio_path ,language='Chinese'):#audio can be wav,ogg,mp3,mp4
        output = self.model.transcribe(audio_path,fp16=False,language=language, temperature=0)
        text = output['text']
        return text
    def auditok(self):
        # 載入 wav 聲音檔案,跳過開頭 2 秒的音訊
        region = auditok.load(self.path, skip=2)
        # 偵測並分割聲音事件，同時繪圖
        # 處理大型檔案
        audio_regions = region.split_and_plot(
            min_dur=0.2,  # 聲音事件的最短長度
            max_dur=4,  # 聲音事件的最長長度
            max_silence=0.3,  # 聲音事件中無訊號最長長度
            energy_threshold=55  # 偵測聲音事件的門檻值
        )
        # 輸出分割聲音事件結果
        for i, r in enumerate(audio_regions):
            # 輸出每段分割音訊的起始與結束時間點
            print("Region {i}: {r.meta.start:.3f}s -- {r.meta.end:.3f}s".format(i=i, r=r))
            # 撥放每段分割音訊
            # r.play(progress_bar=True)
            # 儲存每段分割音訊
            filename = r.save("region_{meta.start:.3f}-{meta.end:.3f}.wav")
            print("儲存為：{}".format(filename))

parser = argparse.ArgumentParser()
parser.add_argument('--load_model_size', type=str, default='medium', help="assign your model size (medium / small / base)")
parser.add_argument('--input_path', type=str, default='./test2.wav', help="assign your wav path file")
opt = parser.parse_args()


if __name__ == '__main__':
    print('init...')
    t1 = time.time()
    W = Whisper(model_name=opt.__dict__['load_model_size'],device_mode='cpu')
    t2 = time.time()
    print(f' load model: , spend {t2-t1} sec')
    text = W.run_whisper(audio_path = opt.__dict__['input_path'] , language = 'English')
    t3 = time.time()
    print(f"model_size:{opt.__dict__['load_model_size']} , pred. spend {t3-t2} sec")
    # sub_wav=W.auditok()
    print(text)
    # python OpenAI_Wisper.py --load_model_size small --input_path ./test1.wav



















