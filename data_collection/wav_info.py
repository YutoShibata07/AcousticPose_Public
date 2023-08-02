import wave
import pyaudio
import os


def printWaveInfo(wf):
    """WAVEファイルの情報を取得"""
    print("チャンネル数:", wf.getnchannels())
    print("サンプル幅:", wf.getsampwidth())
    print("サンプリング周波数:", wf.getframerate())
    print("フレーム数:", wf.getnframes())
    print("パラメータ:", wf.getparams())
    print("長さ（秒）:", float(wf.getnframes()) / wf.getframerate())


if __name__ == "__main__":
    wf = wave.open(os.path.join("data/Mic/210528_001.WAV"), "r")

    printWaveInfo(wf)

    # # ストリームを開く
    # p = pyaudio.PyAudio()
    # stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #                 channels=wf.getnchannels(),
    #                 rate=wf.getframerate(),
    #                 output=True)

    # # チャンク単位でストリームに出力し音声を再生
    # chunk = 1024
    # data = wf.readframes(chunk)
    # while data != '':
    #     stream.write(data)
    #     data = wf.readframes(chunk)
    # stream.close()
    # p.terminate()
