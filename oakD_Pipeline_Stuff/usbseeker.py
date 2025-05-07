import cv2
import depthai as dai
import numpy as np

# Path to your YOLOv8n COCO blob
NN_BLOB_PATH = r"C:/Users/9cdix/Desktop/EVT/ROB-Autonomous/blobs/yolov8n_coco_640x352.blob"

# COCO class labels (80 total)
labelMap = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","pottedplant","bed",
    "diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

def main():
    # 1) Build the processing pipeline
    pipeline = dai.Pipeline()

    # 2) Configure onboard color camera (OAK-D Pro) on CAM_A
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    # Use 1080p resolution supported by the color sensor
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # 3) Resize RGB for YOLO input (640×352 planar BGR)
    manipRgb = pipeline.create(dai.node.ImageManip)
    manipRgb.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manipRgb.initialConfig.setResize(640, 352)
    manipRgb.initialConfig.setKeepAspectRatio(False)
    camRgb.video.link(manipRgb.inputImage)

    # 4) Mono cameras → downsample for stereo depth (use supported 720p)
    monoLeft  = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    leftManip = pipeline.create(dai.node.ImageManip)
    rightManip = pipeline.create(dai.node.ImageManip)
    for m in (leftManip, rightManip):
        m.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        m.initialConfig.setResize(640, 480)
        m.initialConfig.setKeepAspectRatio(False)
    monoLeft.out.link(leftManip.inputImage)
    monoRight.out.link(rightManip.inputImage)

    # 5) Stereo depth aligned to color (CAM_A), output 640×352
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setInputResolution(640, 480)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 352)
    leftManip.out.link(stereo.left)
    rightManip.out.link(stereo.right)

    # 6) Spatial YOLOv8n detection network
    spatialDet = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    spatialDet.setBlobPath(NN_BLOB_PATH)
    spatialDet.setConfidenceThreshold(0.5)
    spatialDet.setNumClasses(len(labelMap))
    spatialDet.setCoordinateSize(4)
    spatialDet.setIouThreshold(0.5)
    spatialDet.setBoundingBoxScaleFactor(0.5)
    spatialDet.setNumInferenceThreads(2)
    manipRgb.out.link(spatialDet.input)
    stereo.depth.link(spatialDet.inputDepth)

    # 7) XLinkOut outputs for host
    xoutRgb     = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    manipRgb.out.link(xoutRgb.input)

    xoutSpatial = pipeline.create(dai.node.XLinkOut)
    xoutSpatial.setStreamName("spatial")
    spatialDet.out.link(xoutSpatial.input)

    # 8) Start USB device and host-side queues
    with dai.Device(pipeline) as device:
        # Use updated intensity APIs
        device.setIrLaserDotProjectorIntensity(200)
        device.setIrFloodLightIntensity(150)

        qRgb     = device.getOutputQueue("rgb",     maxSize=4, blocking=False)
        qSpatial = device.getOutputQueue("spatial", maxSize=4, blocking=False)

        # Create a resizable window and set initial size
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB", 1280, 720)

        while True:
            frame     = qRgb.get().getCvFrame()
            inSpatial = qSpatial.tryGet()

            if inSpatial:
                for det in inSpatial.detections:
                    x1 = int(det.xmin * frame.shape[1])
                    y1 = int(det.ymin * frame.shape[0])
                    x2 = int(det.xmax * frame.shape[1])
                    y2 = int(det.ymax * frame.shape[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cid = int(det.label)
                    name = labelMap[cid] if 0 <= cid < len(labelMap) else f"CLS {cid}"
                    conf = det.confidence
                    depth_m = det.spatialCoordinates.z / 1000.0
                    text = f"{name}: {conf:.2f}, Z: {depth_m:.2f}m"
                    cv2.putText(frame, text, (x1 + 3, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            cv2.imshow("RGB", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
