# test_camera.py - Simple camera test
import cv2

print("Testing cameras...")

for i in range(3):
    print(f"\nTrying camera index {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        print(f"✓ Camera {i} works!")
        ret, frame = cap.read()
        if ret:
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cv2.imshow(f'Camera {i} Test', frame)
            print("  Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"✗ Camera {i} not available")

print("\nTest complete!")