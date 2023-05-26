def formatted_yolo_detection(detection, class_names=None):
    """
    Formats the YOLO detection results.

    Args:
        detection (object): The detection object.
        class_names (dict): Dict of class names by class id.

    Returns:
        list: Formatted detection results.
    """
    formatted_detection = []

    for data in detection.boxes.data.tolist():
        # Get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # Get class id and name
        class_id = int(data[5])
        class_name = class_names[class_id] if class_names is not None else None

        # Add standard format detections
        formatted_detection.append([class_id, class_name, confidence, [xmin, ymin, xmax, ymax]])

    return formatted_detection
