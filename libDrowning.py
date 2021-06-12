import statistics
import cv2

# DROWNING.now_actions / DROWNING.last_actions for current & last frames
class DROWNING:
    def __init__(self, moving_avg = 3 ):
        self.moving_avg = moving_avg
        self.now = []
        self.last = []
        self.now_actions = {}
        self.drowning_hotlist = {}
        self.predrowning_hotlist = {}

    def vh_move_ratio(self, p1, p2):
        x_move = abs(p2[0] - p1[0])
        y_move = abs(p2[1] - p1[1])

        ratio_vh = y_move / x_move

        return ratio_vh

    def punch(self, pdata):
        now = self.now.copy()
        last = self.last.copy()

        now.append(pdata.copy())
        if len(now)>self.moving_avg:
            first = now.pop(0)
            last.append(first)

        if len(last)>self.moving_avg:
            last.pop(0)

        self.now = now.copy()
        self.last = last.copy()

        #set moving list for last datas
        last_moving_list = {}
        for item_id, ldata in enumerate(self.last):

            for ID in ldata:
                if ID in last_moving_list:
                    (poses, heads, bodys, lbodys, ubodys, ious) = last_moving_list[ID]
                else:
                    poses, heads, bodys, lbodys, ubodys, ious = [], [], [], [], [], []

                poses.append(ldata[ID][5])
                (head,body,lbody,ubody) = ldata[ID][4]
                heads.append(head)
                bodys.append(body)
                lbodys.append(lbody)
                ubodys.append(ubody)
                ious.append(ldata[ID][6])

                last_moving_list.update( { ID:(poses, heads, bodys, lbodys, ubodys, ious) } )

        self.movelist_last = last_moving_list
        #print(' test punch', last_moving_list)

        #set moving list for now datas
        now_moving_list = {}
        for item_id, ndata in enumerate(self.now):

            for ID in ndata:
                if ID in now_moving_list:
                    (poses, heads, bodys, lbodys, ubodys, ious) = now_moving_list[ID]
                else:
                    poses, heads, bodys, lbodys, ubodys, ious = [], [], [], [], [], []

                poses.append(ndata[ID][5])
                (head,body,lbody,ubody) = ndata[ID][4]
                heads.append(head)
                bodys.append(body)
                lbodys.append(lbody)
                ubodys.append(ubody)
                ious.append(ndata[ID][6])

                now_moving_list.update( { ID:(poses, heads, bodys, lbodys, ubodys, ious) } )

        #print('now', now)
        #print('last', last)
        #print('-----------------------------------------------------------')
        self.movelist_last = last_moving_list
        self.movelist_now = now_moving_list
        #print(' test punch', now_moving_list)
        self.moving_summarize()

    def avg_boxes(self, boxes):
        counts = len(boxes)
        if not counts>0:
            return None

        i = 0
        xx,yy,ww,hh = 0, 0, 0, 0
        for box in boxes:
            if not len(box)>0: continue

            (x,y,w,h) = box
            xx += x
            yy += y
            ww += w
            hh += h
            i += 1

        if i>0:
            return (int(xx/i), int(yy/i), int(ww/i), int(hh/i))
        else:
            return None

    def moving_summarize(self):
        last_movelist = self.movelist_last
        now_movelist = self.movelist_now

        #print('last_movelist', last_movelist)
        #print('now_movelist', now_movelist)

        last_summarize = {}
        for ID in last_movelist:
            poses = last_movelist[ID][0]
            heads = last_movelist[ID][1]
            bodys = last_movelist[ID][2]
            lbodys = last_movelist[ID][3]
            ubodys = last_movelist[ID][4]
            ious = last_movelist[ID][5]

            last_pose = max(poses, key=poses.count)   #select the max occurrences of pose from last
            #remove none for ious
            ious = [i for i in ious if i]
            if len(ious)>0:
                last_iou = statistics.mean(ious)
            else:
                last_iou = None

            last_head = self.avg_boxes(heads)
            last_body = self.avg_boxes(bodys)
            last_lbody = self.avg_boxes(lbodys)
            last_ubody = self.avg_boxes(ubodys)

            last_summarize.update( { ID: [last_pose, last_head, last_body, last_lbody, last_ubody, last_iou]} )

        self.last_actions = last_summarize

        now_summarize = {}
        for ID in now_movelist:
            poses = now_movelist[ID][0]
            heads = now_movelist[ID][1]
            bodys = now_movelist[ID][2]
            lbodys = now_movelist[ID][3]
            ubodys = now_movelist[ID][4]
            ious = now_movelist[ID][5]

            now_pose = max(poses, key=poses.count)   #select the max occurrences of pose from last
            #remove none in ious
            ious = [i for i in ious if i]
            if len(ious)>0:
                now_iou = statistics.mean(ious)
            else:
                now_iou = None

            now_head = self.avg_boxes(heads)
            now_body = self.avg_boxes(bodys)
            now_lbody = self.avg_boxes(lbodys)
            now_ubody = self.avg_boxes(ubodys)

            now_summarize.update( { ID: [now_pose, now_head, now_body, now_lbody, now_ubody, now_iou]} )

        self.now_actions = now_summarize
        print(now_summarize)

    def detect_drowning(self, img, th_hot_list, drown_sure_frames):
        now_actions = self.now_actions
        last_actions = self.last_actions

        #print('last', last_actions)
        #print('now', now_actions)

        for ID in now_actions:
            now_pose = now_actions[ID][0]
            now_head = now_actions[ID][1]
            now_body = now_actions[ID][2]
            now_lbody = now_actions[ID][3]
            now_ubody = now_actions[ID][4]
            now_iou = now_actions[ID][5]

            if ID in last_actions:
                last_pose = last_actions[ID][0]
                last_head = last_actions[ID][1]
                last_body = last_actions[ID][2]
                last_lbody = last_actions[ID][3]
                last_ubody = last_actions[ID][4]
                last_iou = last_actions[ID][5]

                now_body_centroid = ( now_body[0]+int(now_body[2]/2) , now_body[1]+int(now_body[3]/2) )
                last_body_centroid = ( last_body[0]+int(last_body[2]/2) , last_body[1]+int(last_body[3]/2) )

                print('now_body_centroid, last_body_centroid', (now_body_centroid,last_body_centroid))
                y_movement = now_body_centroid[1] - last_body_centroid[1]
                x_movement = now_body_centroid[0] - last_body_centroid[0]
                ratio_ymove = (abs(y_movement) / now_body[3]) * 100
                ratio_xmove = (abs(x_movement) / now_body[2]) * 100

                print(ID, 'move ratio', ratio_xmove,ratio_ymove)

                hotlist = self.drowning_hotlist
                if now_pose is 0 and (ratio_xmove+ratio_ymove)<th_hot_list:  #register or update the count
                    if ID in hotlist:
                        counts = hotlist[ID][0] + 1
                    else:
                        counts = 0

                    if counts> drown_sure_frames:
                        bcolor = (0,0,255)
                        cv2.putText(img,  'Drowning!', (now_body[0],now_body[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,  bcolor, 2, cv2.LINE_AA)
                    else:
                        bcolor = (0,255,255)

                    cv2.rectangle(img, (now_body[0],now_body[1]), (now_body[0]+now_body[2], now_body[1]+now_body[3]), bcolor, 1)
                    hotlist.update( { ID:[counts, now_body] })

                elif ID in hotlist:
                    hotlist.pop(ID)

        return img
