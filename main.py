import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=2)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect:
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Draw mode on
    # hands, img = detector.findHands(img, draw=False)  # Draw mode off

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # list of 21 Landmark positions of the first hand
        l1, _1, _1 = detector.findDistance(lmList1[8], lmList1[12], img)
        # print(l1)
        if l1 < 30:
            cursor = lmList1[8]  # index finger tip landmark
            # call the update here
            for rect in rectList:
                rect.update(cursor)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # list of 21 Landmark positions of the second hand
            l2, _2, _2 = detector.findDistance(lmList2[8], lmList2[12], img)
            # print(l2)
            if l2 < 30:
                cursor = lmList2[8]  # index finger tip landmark
                # call the update here
                for rect in rectList:
                    rect.update(cursor)

        # Draw solid
        # for rect in rectList:
        #     cx, cy = rect.posCenter
        #     w, h = rect.size
        #     cv2.rectangle(img, (cx - w // 2, cy - h // 2),
        #                   (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        #     cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

        # Draw Transparency
        imgNew = np.zeros_like(img, np.uint8)
        for rect in rectList:
            cx, cy = rect.posCenter
            w, h = rect.size
            cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                          (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
            cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

        out = img.copy()
        alpha = 0.5
        mask = imgNew.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    if hands:
        cv2.imshow("Image", out)
        cv2.waitKey(1)
    else:
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    # if hands:
    #     # Hand 1
    #     hand1 = hands[0]
    #     lmList1 = hand1["lmList"]  # list of 21 Landmark positions of the first hand
    #     bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h of the first hand
    #     centerPoint1 = hand1["center"]  # center of the hand cx, cy
    #     handType1 = hand1["type"]  # Hand type (Left or Right)
    #
    # if len(hands) == 2:
    #     # Hand 2
    #     hand2 = hands[1]
    #     lmList2 = hand2["lmList"]  # list of 21 Landmark positions of the second hand
    #     bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h of the second hand
    #     centerPoint2 = hand2["center"]  # center of the hand cx, cy
    #     handType2 = hand2["type"]  # Hand type (Left or Right)

