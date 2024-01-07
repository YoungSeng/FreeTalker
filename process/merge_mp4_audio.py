# from moviepy.editor import *
# import os
# import subprocess
# import argparse
# import pdb
#
#
# # def add_audio_to_video(video_file, audio_tag, source_BEAT_path, output_video_file, merge_segment, i=0):
# #     # 加载视频和音频文件
# #     video = VideoFileClip(video_file)
# #     # 计算帧范围对应的时间
# #     fps = video.fps
# #     total_frames = video.reader.nframes
# #
# #     final_video = None
# #     # for i in range(len(audio_tag)):
# #     current_video = final_video if final_video is not None else video
# #
# #     speaker, audio_file, start_frame, end_frame = audio_tag[i]
# #
# #     audio_file = os.path.join(source_BEAT_path, speaker, audio_file + '.wav')
# #
# #     audio = AudioFileClip(audio_file, fps=16000)
# #
# #     start_time = merge_segment[i][0] / fps
# #     end_time = merge_segment[i][1] / fps
# #
# #     audio_clip = audio.subclip(start_frame / fps, end_frame / fps)
# #
# #     assert audio_clip.duration == end_time - start_time, f"音频时长{audio_clip.duration}与视频时长{end_time - start_time}不一致"
# #
# #     # 在指定时间范围内设置音频
# #     video_with_new_audio = current_video.subclip(start_time, end_time).set_audio(audio_clip)
# #
# #     print("fps", fps, "total_frames", total_frames, "start_time", start_time, "end_time", end_time, "audio start frame", start_frame,
# #           "audio end frame", end_frame, "audio_clip.duration", audio_clip.duration)
# #
# #     # 将修改过的视频片段与原始视频合并
# #     final_video = concatenate_videoclips([current_video.subclip(0, start_time), video_with_new_audio, current_video.subclip(end_time)])     # , total_frames / fps
# #
# #     # 保存最终的视频文件
# #     final_video.write_videofile(output_video_file)
# #
# #     # 关闭视频和音频文件
# #     video.close()
# #     final_video.close()
# #     audio.close()
# #     audio_clip.close()
# #
# #     return output_video_file
#
# from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
#
# import subprocess
#
# def convert_wav_to_mp3(wav_file, mp3_file):
#     command = f"ffmpeg -i {wav_file} {mp3_file}"
#     subprocess.run(command, shell=True, check=True)
#
#
# def add_audio_to_video(video_file, audio_tags, audio_file, output_video_file, merge_segments):
#     # 加载视频文件
#     video = VideoFileClip(video_file)
#
#     # 计算帧范围对应的时间
#     fps = video.fps
#
#     audio_clips = []
#
#     for i in range(len(audio_tags)):
#         speaker, _, start_frame, end_frame = audio_tags[i]
#         audio = AudioFileClip(audio_file, fps=16000)
#         start_time = merge_segments[i][0] / fps
#         end_time = merge_segments[i][1] / fps
#
#         audio_clip = audio.subclip(start_frame / fps, end_frame / fps)
#         audio_clip = audio_clip.set_start(start_time)
#
#         audio_clips.append(audio_clip)
#
#         # 关闭音频剪辑
#         audio.close()
#         audio = None
#
#     # 创建一个组合音频剪辑
#     composite_audio = CompositeAudioClip(audio_clips)
#
#     # 将音频添加到视频中
#     final_video = video.set_audio(composite_audio)
#
#     # 保存最终的视频文件
#     final_video.write_videofile(output_video_file)
#
#     # 关闭视频文件
#     video.close()
#     final_video.close()
#
#     return output_video_file
#
#
#
# if __name__ == '__main__':
#     '''
#     python -m process.merge_mp4_audio
#     '''
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--video_file", type=str, required=True, help='stick figure mp4 file to be rendered.')
#     # parser.add_argument("--source_BEAT_path", type=str)
#     # parser.add_argument("--output_video_file", type=str)
#     # parser.add_argument("--audio_tag", type=list, nargs='+')
#     # parser.add_argument("--merge_segment", type=list, nargs='+')
#     # params = parser.parse_args()
#
#     # video_file = params.video_file
#     # source_BEAT_path = params.source_BEAT_path
#     # output_video_file = params.output_video_file
#     # audio_tag = params.audio_tag
#     # merge_segment = params.merge_segment
#
#     # 示例用法
#     video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/positions_real_123456789.mp4"
#     # source_BEAT_path = "/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/"
#     output_video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/positions_real_123456789-with-audio.mp4"
#     audio_tag = [['2', '2_scott_0_55_55', 0, 105], ['2', '2_scott_0_55_55', 105, 105*2], ['2', '2_scott_0_55_55', 105*2, 105*3]]
#     merge_segment = [[180, 180+105], [180+105, 180+105*2], [180+105*2, 180+105*3]]
#
#     # output_video = add_audio_to_video(video_file, audio_tag, source_BEAT_path, output_video_file, merge_segment, i=0)
#
#
#     # video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/positions_real_123456789-with-audio.mp4"
#     # output_video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/positions_real_123456789-with-audio-2.mp4"
#     # output_video = add_audio_to_video(video_file, audio_tag, source_BEAT_path, output_video_file, merge_segment, i=1)
#     # output_video = add_audio_to_video(video_file, audio_tag, source_BEAT_path, output_video_file, merge_segment, i=2)
#
#     # for i in range(len(audio_tag)):
#     #     output_video_file = add_audio_to_video(video_file, audio_tag, source_BEAT_path, output_video_file,
#     #                                            merge_segment, i)
#     #     video_file = output_video_file
#     #
#     # print(f"输出视频文件: {output_video_file}")
#
#     wav_file = '/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/2/2_scott_0_55_55.wav'
#     mp3_file = '/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/2_scott_0_55_55.mp3'
#
#     convert_wav_to_mp3(wav_file, mp3_file)
#     output_video = add_audio_to_video(video_file, audio_tag, mp3_file, output_video_file, merge_segment)
#     print(f"输出视频文件: {output_video}")


from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.editor import AudioFileClip
import os
from io import BytesIO
import tempfile
import argparse


def add_audio_to_video_pydub(video_file, audio_tags, source_BEAT_path, output_video_file, merge_segments):
    # 加载视频文件
    video = VideoFileClip(video_file)

    # 计算帧范围对应的时间
    fps = video.fps

    # 创建一个空的音频片段
    final_audio = AudioSegment.silent(duration=video.duration * 1000)

    for i in range(len(audio_tags)):
        speaker, audio_file, start_frame, end_frame = audio_tags[i]
        # audio_file = os.path.join(source_BEAT_path, speaker, audio_file + '.mp3')
        audio_file = os.path.join(source_BEAT_path, speaker, audio_file + '.wav')
        # audio_file = '/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/2_scott_0_55_55.mp3'
        # 加载音频文件
        # audio = AudioSegment.from_mp3(audio_file)
        audio = AudioSegment.from_wav(audio_file)

        # 截取音频文件
        audio_clip = audio[start_frame * 1000 // fps:end_frame * 1000 // fps]

        # 将音频片段添加到最终音频中
        start_time = merge_segments[i][0] * 1000 // fps
        final_audio = final_audio.overlay(audio_clip, position=start_time)

    # 将音频添加到视频中
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        final_audio.export(temp_audio_file.name, format="mp3")
        video.audio = AudioFileClip(temp_audio_file.name)

    # 保存最终的视频文件
    video.write_videofile(output_video_file)

    # 关闭视频文件
    video.close()

    return output_video_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str,
                        default="/ceph/hdd/yangsc21/Python/mdm/save/my_v3_0/model001000000-s2/positions_real_1234567_1.0.mp4")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # 示例用法
    # video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/positions_real_123456789.mp4"
    # source_BEAT_path = "/apdcephfs/share_1290939/new_data/BEAT/beat_english_v0.2.1/"
    # output_video_file = "/apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/model000600000/positions_real_123456789-with-audio.mp4"
    # audio_tags = [['2', '2_scott_0_55_55', 0, 105], ['2', '2_scott_0_55_55', 105, 105*2], ['2', '2_scott_0_55_55', 105*2, 105*3]]
    # merge_segments = [[180, 180+105], [180+105, 180+105*2], [180+105*2, 180+105*3]]
    #
    # output_video = add_audio_to_video_pydub(video_file, audio_tags, source_BEAT_path, output_video_file, merge_segments)
    # print(f"输出视频文件: {output_video}")

    # 'python', '-m', 'mdm_motion2smpl', '--input', source_npy, '--output', target_path
    # python -m mdm_motion2smpl --input /apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/result_rec_123456789.npy --output /apdcephfs/private_yyyyyyyang/code/mdm/save/my_v2_0/

    args = get_args()

    audio_tags = [['2', '2_scott_0_14_14', 0, 110 * 2],
                  ['2', '2_scott_0_37_37', 0, 110 * 2]]

    # audio_tags = [['4', '4_lawrence_0_19_19', 0, 110 * 2],
    #               ['4', '4_lawrence_0_35_35', 0, 110 * 2]]
    merge_segments = [[90, 90 + 110 * 2], [380, 380 + 110 * 2]]

    video_file = args.video_file
    output_video_file = video_file.replace(".mp4", "-with-audio.mp4")
    source_BEAT_path = "./data/BEAT/beat_english_v0.2.1/"

    output_video = add_audio_to_video_pydub(video_file, audio_tags, source_BEAT_path, output_video_file, merge_segments)
