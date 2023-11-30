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
                if self.print:
                    logger.debug('%s' % msg)
                try:
                    meta = json.loads(msg)
                    if 'HailoROI' in meta:
                        if 'SubObjects' in meta['HailoROI']:
                            subobjs = meta['HailoROI']['SubObjects']
                            for obj in subobjs:
                                if  'HailoDetection' in obj:
                                    if self.print:
                                        logger.info('==================')                                        

                                    det = obj['HailoDetection']
                                    if self.print:
                                        logger.debug('Detection, type: %s: conf: %f, x,y:(%f, %f), w,h:(%f, %f)' % (
                                            det['label'],
                                            det['confidence'],
                                            det['HailoBBox']['xmin'],
                                            det['HailoBBox']['ymin'],
                                            det['HailoBBox']['width'],
                                            det['HailoBBox']['height']))

                                    bb = {}
                                    bb['xmin'] = det['HailoBBox']['xmin']
                                    bb['xmax'] = det['HailoBBox']['xmin'] + det['HailoBBox']['width']
                                    bb['ymin'] = det['HailoBBox']['ymin']
                                    bb['ymax'] = det['HailoBBox']['ymin'] + det['HailoBBox']['height']

                                    if 'SubObjects' in det:
                                        item = {}
                                        for detobj in det['SubObjects']:
                                            for key in detobj.keys():
                                                if self.print:
                                                    logger.info('Object type: %s' % key)

                                                # Face recognition    
                                                if key == 'HailoClassification':
                                                    if detobj[key]['classification_type'] == 'recognition_result':
                                                        #logger.debug('face recognized as: %s' % detobj[key]['label'])
                                                        item['type'] = 'face_recog'
                                                        item['bb'] = bb
                                                        item['conf'] = det['confidence']
                                                        item['id'] = detobj[key]['label'].rstrip(string.digits)
                                                
                                                # Tracked face
                                                elif key == 'HailoUniqueID' and det['label'] == 'person':
                                                    item['type'] = 'person'
                                                    item['bb'] = bb
                                                    item['conf'] = det['confidence']

                                                    # Re-id ID
                                                    if detobj[key]['mode'] == 1:
                                                        item['reid_id'] = detobj[key]['unique_id'] 
                                                        logger.info('reid id: %s' % detobj[key]['unique_id'])
                                                
                                                    # Tracker ID
                                                    elif detobj[key]['mode'] == 0:
                                                        item['tracker_id'] = detobj[key]['unique_id']
                                                        logger.info('tracker id: %s' % detobj[key]['unique_id'])

                                                    
                                        if len(item.keys()) > 0:
                                            out.append(item)
                except e:
                    logger.info("error parsing hailo meta")
                    pass

            except zmq.Again as e:
                pass

        return out
