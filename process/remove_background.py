import os
from rembg import remove
# from PIL import Image

def remove_background(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)

            with open(input_file, 'rb') as img:
                img_data = img.read()
                output = remove(img_data)

            with open(output_file, 'wb') as out:
                out.write(output)

            print(f'已成功处理：{filename}')

input_folder = r"C:\Users\94086\Seafile\科研\Tencent\ICASSP\guidance\pics_nobg\new"
output_folder = input_folder + '_nobg'

remove_background(input_folder, output_folder)