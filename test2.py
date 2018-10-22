'''

  Created by irving on 19/10/18

'''

import cv2
import skvideo.io


cap = cv2.VideoCapture('/home/irving/IMG_4231.MOV')
ret, frame = cap.read()
width, height, channel = frame.shape
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# writer = cv2.VideoWriter('rotate.avi', fourcc, 20, (width, height))

writer = skvideo.io.FFmpegWriter('rotate.avi', inputdict={'-r': str(30)},
                                 outputdict={
                                     '-vcodec': 'libx264', '-b': '300000000'
                                 }
                                 )
num_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    num_frame += 1
    v_frame = cv2.flip(frame, 0)[100:, 400:-100, :]
    cv2.imshow('Demo', v_frame)
    cv2.imwrite('/mnt/sda/people_tracking/tmp/' + str(num_frame).zfill(6) + '.jpg', v_frame)
    writer.writeFrame(v_frame[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.close()
# writer.release()
cv2.destroyAllWindows()

