import cv2
import depthai as dai
 
def main(): # Create pipeline and define device
    
    pipeline = dai.Pipeline()
 
    # ---------- Pipeline Setup ----------
    camRgb = pipeline.createColorCamera()
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)  # Recommended for best performance
    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.video.link(xoutRgb.input)
    # ---------- Ethernet Connection Setup ----------
    device_ip = "169.254.1.222"  
    device_info = dai.DeviceInfo(device_ip)
 
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    
    
    try: # Select device and freeze on the first frame (take a screenshot type shii)
        with dai.Device(pipeline, device_info) as device:
            print("Connected to device via Ethernet!")
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
 
            # --- Grab first frame for ROI selection ---
            first_frame = None
            while first_frame is None:
                inRgb = qRgb.get()  
                first_frame = inRgb.getCvFrame()
    # Selects object for tracking and closes tracking window
            # Let user draw a box around the object to track
            bbox = cv2.selectROI("RGB", first_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("ROI selector")
 
            # Create and initialize the CSRT tracker
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first_frame, bbox)
        
            # ---------- Main Loop with Tracking ----------
            while True: # Stores object for tracking and overlays a green rectangle around it
                            # if tracking fails a message gets displayed and you reselect
                            # if tracking drifts you can reselect the ROI by pressing 'r'
                if cv2.getWindowProperty("RGB", cv2.WND_PROP_VISIBLE) < 1:
                    break
 
                inRgb = qRgb.get()  
                frame = inRgb.getCvFrame()
 
                # Update tracker and get updated bounding box
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, bbox)
                    # Draw a green rectangle around the tracked object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
                else:
                    # Tracking failure message
                    cv2.putText(frame, "Tracking failure detected", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
 
                cv2.imshow("RGB", frame)
 
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Re‑select ROI if tracking drifts
                    bbox = cv2.selectROI("RGB", frame, fromCenter=False, showCrosshair=True)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
    except KeyboardInterrupt:
        print("Exiting on keyboard interrupt.")
    except Exception as e:
        print("An error occurred:", e)
    finally:
        cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()