import cv2
import argparse # for try to make webcam configurable 

from ultralytics import YOLO
import supervision as sv # for draw bounding boxes on the frame
import numpy as np

margin = 0.25 
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5 - margin, 0],
    [0.5 - margin, 1],
    [0, 1]
])

ZONE_POLYGON_ = np.array([
    [0.5 + margin, 0],
    [1, 0],
    [1, 1],
    [0.5 + margin, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=3,
        text_scale=1      
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    # A class for annotating a polygon-shaped zone within a frame with a count of detected objects.
    zone = sv.PolygonZone(
        polygon=zone_polygon,
        frame_resolution_wh=tuple(args.webcam_resolution) 
    )
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=10,
        text_thickness=6,
        text_scale=4
    )

    zone_polygon_ = (ZONE_POLYGON_ * np.array(args.webcam_resolution)).astype(int)
    zone_ = sv.PolygonZone(
        polygon=zone_polygon_,
        frame_resolution_wh=tuple(args.webcam_resolution) 
    )
    zone_annotator_ = sv.PolygonZoneAnnotator(
        zone=zone_, 
        color=sv.Color.blue(),
        thickness=10,
        text_thickness=6,
        text_scale=4
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to read frame from the webcam.")
                break
            # Agnostic NMS (Non-Maximum Suppression): Eliminating double detections
            result = model(frame, agnostic_nms=True)[0]
            detections = sv.Detections.from_yolov8(result)
            # let filter person out
            detections = detections[detections.class_id != 0]
            
            labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _ 
                in detections
            ]

            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )

            zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)

            zone_.trigger(detections=detections)
            frame = zone_annotator_.annotate(scene=frame)

            cv2.imshow("yolov8", frame)


            # ASCII table 'escape' key : 27    
            if cv2.waitKey(30) == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
