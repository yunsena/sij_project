import io
import socket
import struct
import time
from PIL import Image
import os
import time

def get_time():
    now = time.localtime()
    s = "%04d%02d%02d-%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

# edgesdf
# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('192.168.0.187', 80))
server_socket.listen(0)
#test
# Accept a single connection and make a file-like object out of it
print('server start')

connection = server_socket.accept()[0].makefile('rb')
try:
    print('client accept')
    cnt = 0
    #folder_num = 1
    dir_name = get_time()
    os.mkdir('/home/jeong/fall_detection_online/test/rec/'+ str(dir_name) +'/')
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        image = Image.open(image_stream)
        print('Image is %dx%d' % image.size)        
        image_name = '/home/jeong/fall_detection_online/test/rec/'+ str(dir_name) +'/image' + str(cnt+1) + '.jpeg'
        image.save(image_name)
        cnt = cnt + 1
        image.verify()
        print('Image is verified')
        if cnt % 150 == 0:
             #folder_num = folder_num + 1
             dir_name = get_time()
             os.mkdir('/home/jeong/fall_detection_online/test/rec/'+ str(dir_name) +'/')

finally:
    connection.close()
    server_socket.close()
