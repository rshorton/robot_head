import zmq
import json
import string

# Receives meta data generated from a Hailo AI pipeline and published
# from the pipeline using an ZMQ publisher.

# Since the current use case implemented by the Hailo pipeline for Elsabot
# is for facial recognition, this code only parsed-out the meta data
# specific to facial recog.

class hailo_zmq_meta_sink:
    def __init__(self, port):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)

        self.is_open = False
        self.print = False

    def open(self):
        self.socket.connect('tcp://127.0.0.1:' + str(self.port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.is_open = True

    def close(self):
        self.socket.close()

    def read(self, logger):
        out = []
        for i in range(10):
            try:
                msg = self.socket.recv(flags=zmq.NOBLOCK)
                msg = msg.decode()
                #print("%s" % (msg))
                try:
                    meta = json.loads(msg)
                    if 'HailoROI' in meta:
                        if 'SubObjects' in meta['HailoROI']:
                            subobjs = meta['HailoROI']['SubObjects']
                            for obj in subobjs:
                                if  'HailoDetection' in obj:
                                    if self.print:
                                        print('==================')                                        

                                    det = obj['HailoDetection']
                                    if self.print:
                                        print('Detection, type: %s: conf: %f, x,y:(%f, %f), w,h:(%f, %f)' % (
                                        det['label'],
                                        det['confidence'],
                                        det['HailoBBox']['xmin'],
                                        det['HailoBBox']['ymin'],
                                        det['HailoBBox']['width'],
                                        det['HailoBBox']['height']))

                                    if 'SubObjects' in det:
                                        for detobj in det['SubObjects']:
                                            for key in detobj.keys():
                                                if self.print:
                                                    print('Object type: %s' % key)
                                                if key == 'HailoClassification':
                                                    if detobj[key]['classification_type'] == 'recognition_result':
                                                        #print('face recognized as: %s' % detobj[key]['label'])
                                                        item = {}
                                                        item['type'] = 'face_recog';
                                                        item['bb'] = det['HailoBBox']
                                                        item['conf'] = det['confidence']
                                                        item['label'] = detobj[key]['label'].rstrip(string.digits)
                                                        out.append(item)
                except:
                    pass

            except zmq.Again as e:
                pass

        return out
