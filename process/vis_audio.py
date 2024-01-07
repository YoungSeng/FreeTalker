import librosa
import librosa.display
import matplotlib.pyplot as plt

# def visualize_audio_waveform(audio_file, output_file, num_frames=305, sample_rate=16000, fps=20):
#     # 加载音频文件
#     audio_data, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
#
#     # 计算每帧的音频样本数
#     samples_per_frame = sample_rate // fps
#
#     # 只保留前 num_frames 帧的音频数据
#     audio_data = audio_data[:num_frames * samples_per_frame]
#
#     # 创建一个新的 matplotlib 图像
#     plt.figure()
#
#     # 绘制音频波形
#     librosa.display.waveshow(audio_data, sr=sample_rate, alpha=1, color='k')
#
#     # 设置图像标题和轴标签
#     plt.title('Audio Waveform')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#
#     # 保存结果为 PDF 文件
#     plt.savefig(output_file, format='pdf')
#
#     # 关闭图像
#     plt.close()

def visualize_audio_waveform(audio_file, output_file, num_frames=305, sample_rate=16000, fps=20, dpi=300):
    # 加载音频文件
    audio_data, _ = librosa.load(audio_file, sr=sample_rate, mono=True)

    # 计算每帧的音频样本数
    samples_per_frame = sample_rate // fps

    # 只保留前 num_frames 帧的音频数据
    audio_data = audio_data[:num_frames * samples_per_frame]

    # 创建一个新的 matplotlib 图像
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)

    # 绘制音频波形
    img = librosa.display.waveshow(audio_data, sr=sample_rate, alpha=1, color='k', ax=ax)

    # 移除坐标轴和标题
    ax.axis('off')

    # 保存结果为 PNG 文件
    plt.savefig(output_file, format='png', bbox_inches='tight', pad_inches=0)

    # 关闭图像
    plt.close()

audio_file = r'C:\Users\94086\Seafile\科研\Tencent\SMPL\my_v2_0\60w\123456789\2_scott_0_55_55.mp3'
output_file = r'C:\Users\94086\Seafile\科研\Tencent\SMPL\my_v2_0\60w\123456789\audio_waveform.png'

visualize_audio_waveform(audio_file, output_file)