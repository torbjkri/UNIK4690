import cv2

def projectImage(img1, img2, homography, thresh = 10, maxval = 255):

    w2, h2 = img1.shape
    img2_warped = cv2.warpPerspective(img2, homography, dsize=(h2, w2))

    ret, mask = cv2.threshold(img2_warped, thresh, maxval, cv2.THRESH_BINARY)

    inv = cv2.bitwise_not(mask)
    a = inv.shape
    b = mask.shape
    c = img1.shape
    d = img2.shape
    print(a,b,c,d)
    img1_bg = cv2.bitwise_and(img1, inv)
    img2_fg = cv2.bitwise_and(img2_warped, mask)

    img_12 = cv2.add(img1_bg, img2_fg)

    return img_12