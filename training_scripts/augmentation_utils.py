import numpy as np
import cv2
import tensorflow as tf


def reduce_mean_mse_with_exp_weighting(y_hat,y_target,exp_factor=0):
    assert exp_factor >= 0.0
    weights = tf.exp(tf.abs(y_target) * exp_factor)
    error = y_hat-y_target
    return tf.reduce_sum(weights * tf.square(error)) / tf.reduce_sum(weights)


def adjust_gamma(images, gamma=1.0):
    return np.power(images,gamma)

def draw_shadow(img,thickness=10,blur=3,angle=np.pi/2,offset_x=0,offset_y=0,gamma=1.5):
    mask = np.zeros([img.shape[0],img.shape[1],1])

    r = 200
    point1 = (int(r*np.cos(angle)+offset_x+img.shape[1]/2),int(r*np.sin(angle)+offset_y+img.shape[0]/2))
    point2 = (int(-r*np.cos(angle)+offset_x+img.shape[1]/2),int(-r*np.sin(angle)+offset_y+img.shape[0]/2))
    cv2.line(mask,point1,point2,(1.0),thickness)
    mask = cv2.GaussianBlur(mask,(blur,blur),0)
    mask = mask.reshape([img.shape[0],img.shape[1],1])

    img2 = adjust_gamma(np.copy(img),gamma)
    img_merged = mask * img2 + (1.0-mask) * img
    return img_merged

if __name__ == "__main__":
    img = cv2.imread('images/crop_00.png').astype(np.float32)/255.0
    print("img shape: ",str(img.shape))
    cv2.imwrite('images/save.png',img)
    noise_table = [0.1,0.15,0.2]
    for i in range(len(noise_table)):
        img_noise =  np.clip(img +  np.random.normal(loc=0,scale=noise_table[i],size=img.shape),0,1)
        cv2.imwrite('images/nosie_{:02d}.png'.format(i),img_noise*255)

    # mask = np.zeros([img.shape[0],img.shape[1],1])

    # cv2.line(mask,(0,0),(250,250),(1.0),13,lineType=cv2.LINE_AA)
    # mask = cv2.GaussianBlur(mask,(3,3),0)

    # cv2.line(mask,(50,0),(300,250),(1.0),13,lineType=cv2.LINE_AA)

    # img2 = adjust_gamma(np.copy(img),1.5)

    # mask = mask.reshape([img.shape[0],img.shape[1],1])
    # img_merged = mask * img2 + (1.0-mask) * img

    # img_merged = draw_shadow(img,thickness=40,blur=3,angle=0.234,offset=-50,gamma=0.7)
    # cv2.imwrite("images/shadowed_1.png",np.clip(255*img_merged,0,255))
    # for i in range(20):
    #     thickness = np.random.randint(10,100)
    #     kernel_sizes = [3,5,7]
    #     blur = kernel_sizes[np.random.randint(0,len(kernel_sizes))]
    #     angle = np.random.uniform(low=0,high=np.pi)
    #     # angle = np.random.triangular(left=-np.pi/2,mode=0,right=np.pi/2)
    #     offset_x = np.random.randint(-100,100)
    #     offset_y = np.random.randint(-30,30)
    #     gamma = np.random.uniform(1,2)
    #     do_darken = np.random.rand()>0.33 # 2/3 darker, 1/3 lighter
    #     if(not do_darken):
    #         print("light")
    #         gamma = 1.0/gamma
    #     else:
    #         print("dark")
    #     img_merged = draw_shadow(img,thickness,blur,angle,offset_x,offset_y,gamma)
    #     cv2.imwrite("images/shadowed_{:02d}.png".format(i),np.clip(255*img_merged,0,255))

    # cv2.imshow('image',img_merged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()