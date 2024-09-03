from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication
from PyQt5.QtGui import QPixmap, QTransform, QMouseEvent, QPainter, QPen, QFont, QColor
from PyQt5.QtCore import Qt, QPointF, QSize, QPoint, QTimer
import time, os
import utilities as util
import csv_loader

# For popup TextInputDialog
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton


class ImageCanvas(QGraphicsView):
    def __init__(self, map_image_file=None,output_dir=None):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # Set Window Size
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        # Set window size relative to display size (e.g., 50% of screen width and height)
        width = int(screen_size.width() * 0.7)
        height = int(screen_size.height() * 0.7)
        self.resize(QSize(width, height))
        
        # Verify output dir
        if output_dir is None:
            self.output_dir = util.select_directory()
            if self.output_dir is None:
                print(f"No output directory selected, quitting.")
                exit()
        else:
            self.output_dir = output_dir
        print(f"Output Directory: {self.output_dir}")

        # Add an image
        if map_image_file is None:
            map_image_file = util.open_image_dialog('./data/example')
            if map_image_file is None:
                print(f"No image selected, quitting.")
                exit()
        print(f"Opening image file: {map_image_file}")
        pixmap = QPixmap(map_image_file)
        item = QGraphicsPixmapItem(pixmap)
        item.setOpacity(1.0)  # Set alpha
        item.setTransform(QTransform().rotate(0))  # Set rotation
        self.scene.addItem(item)
        
        # Set up zoom and pan
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)  # Initially, no drag mode
        self.setCursor(Qt.ArrowCursor)

        # Add data and points
        self.data = []
        self.pt_label = None
        self.current_index = 0

        # Add FPS Tracker
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0

        # Setup a timer to update FPS every second
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update FPS every second
        
    def update_fps(self):
        # Calculate and update FPS
        current_time = time.time()
        elapsed_time = current_time - self.last_frame_time
        self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        self.frame_count = 0
        self.last_frame_time = current_time
        

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        super().wheelEvent(event)
        self.viewport().update()  # Trigger paintEvent when resizing  

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Enable panning
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)
        # Record the mouse position on the canvas
        self.mouse_down_pos = event.pos()
        super().mousePressEvent(event)
        self.viewport().update()  # Trigger paintEvent when resizing  

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.viewport().update()  # Trigger paintEvent when panning (if panning is handled via mouse move)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Stop panning
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.ArrowCursor)
            self.mouse_up_pos = event.pos()
            # print(f"Mouse released at: {self.mouse_up_pos}")
            delta = self.mouse_up_pos - self.mouse_down_pos
            
            thresh = 1
            if abs(delta.x()) <= thresh and abs(delta.y()) <= thresh:
                # MOUSE IS CLICKED
                self.mouse_scene_pos = self.mapToScene(event.pos())
                print(f"Mouse clicked at: ({self.mouse_scene_pos.x():.1f},{self.mouse_scene_pos.y():.1f})")
                self.set_reference_point()
            else:
                print(f"Panned: ({delta.x()},{delta.y()})")
        super().mouseReleaseEvent(event)
        self.viewport().update()  # Trigger paintEvent when resizing   

    def set_reference_point(self):
        x = int(self.mouse_scene_pos.x())
        y = int(self.mouse_scene_pos.y())
        if self.pt_label is None:
                self.prompt_for_text()
                if self.pt_label is None:
                    print(f"No point label was set. Nothing added.")
                    return
                self.data.append(dict())
        print("Label: {}     Pos: ({},{})".format(self.pt_label,x,y))
        self.data[self.current_index]['label'] = self.pt_label
        self.data[self.current_index]['x'] = x
        self.data[self.current_index]['y'] = y

    def prompt_for_text(self):
        dialog = TextInputDialog(self,str(len(self.data)+1))  # Create the dialog
        result = dialog.get_text()  # Show the dialog and get the input
        if result:  # Check if input was provided
            print(f"User input: {result}")  # Example: Use the returned text
            self.pt_label = result

    def paintEvent(self, event):
        self.frame_count += 1
        # First, let QGraphicsView paint the scene and everything else
        super().paintEvent(event)
        
        # Now overlay the text
        painter = QPainter(self.viewport())
        
        
        # Assemble the status text string to draw
        if len(self.data) <= self.current_index:
                current_key = None
                current_val = None
        else:
            current_key = self.data[self.current_index]['label']
            current_val = [self.data[self.current_index]['x'],self.data[self.current_index]['y']]
        status_str = "{}/{} - {}: {}".format(self.current_index+1,len(self.data),current_key,current_val)
        text_position = QPointF(10, 30)  # Offset from the top-left corner
        shadow_position = QPointF(14, 34)
        # Customize font, color, and position
        # NOTE: The pens don't seem to work as intended, shadow_position is a backup solution for text outline
        font = QFont("Arial", 18, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(Qt.black, 3)) 
        # Draw the text
        painter.drawText(shadow_position, status_str)
        painter.setPen(QPen(Qt.white,1)) 
        painter.drawText(text_position, status_str)
        
        # Draw FPS
        painter.setFont(QFont("Arial", 12, QFont.Normal))
        painter.setPen(QPen(Qt.black, 3)) 
        fps_position = QPointF(14, 60)
        painter.drawText(fps_position, f"FPS: {int(self.fps)}")

        # Draw all reference points
        painter.setFont(QFont("Arial", 14, QFont.Normal))
        painter.setPen(QPen(Qt.red,3)) 
        for item in self.data:
            scene_pos = QPointF(item['x'],item['y'])
            viewport_point = self.mapFromScene(scene_pos)
            x = int(viewport_point.x())
            y = int(viewport_point.y())
            label = item["label"]
            r = 10
            # Draw the point (cross with radius r)
            painter.drawLine(x,y-r,x,y+r)
            painter.drawLine(x-r,y,x+r,y)

            # Draw the label next to the point
            offset = QPoint(5,-5) # Adjust position as needed
            viewport_point += offset
            painter.drawText(viewport_point, label)  
        
        # End the painter
        painter.end()

    def keyPressEvent(self, event):
        # Handle different key presses
        if event.key() == Qt.Key_Q:
            QApplication.quit()  # Quit the application when Q is pressed
        
        elif event.key() == Qt.Key_X: # EXPORT CSV
            csv_path = csv_loader.save_csv_dialog(self.data,self.output_dir)
            im_file = os.path.join(os.path.dirname(csv_path),os.path.basename(csv_path)[:-4]+'.png')
            #TODO: Save Image File
            # print(f"IM FILE: {im_file}")
            # cv2.imwrite(im_file,draw_img)
            print(f"Saved {im_file}")
        elif event.key() == Qt.Key_Z: # IMPORT CSV
            csv_path = csv_loader.open_csv_dialog(self.output_dir)
            if csv_path is not None:
                headers, self.data = csv_loader.read_csv_points(csv_path)
                self.data = csv_loader.fix_data_types(self.data,set_str=['label'],set_float=['x','y'])
                self.current_index = len(self.data)
        elif event.key() == Qt.Key_1: # Previous Reference Point
            self.current_index = (self.current_index-1)%(len(self.data)+1)
            if len(self.data) <= self.current_index:
                self.pt_label = None
            else:
                self.pt_label = self.data[self.current_index]['label']
            print(f"Index [{self.current_index}] - Label: {self.pt_label}")
        elif event.key() == Qt.Key_2: # Next Reference Point
            self.current_index = (self.current_index+1)%(len(self.data)+1)
            if len(self.data) <= self.current_index:
                self.pt_label = None
            else:
                self.pt_label = self.data[self.current_index]['label']
            print(f"Index [{self.current_index}] - Label: {self.pt_label}")
        elif event.key() == Qt.Key_3: # Remove Reference Point
            if len(self.data) <= self.current_index:
                pass
                print("Nothing removed.")
            else:
                popped = self.data.pop(self.current_index)
                self.current_index = len(self.data)
                self.pt_label = None
                print(f"Removed Index {self.current_index}: {popped}")
        else:
            super().keyPressEvent(event)  # Call the base class method for unhandled keys


class TextInputDialog(QDialog):
    def __init__(self, parent=None, default_text=""):
        super().__init__(parent)
        self.setWindowTitle("Input Text")
        self.setFixedSize(300, 150)

        layout = QVBoxLayout(self)
        
        # Label
        self.label = QLabel("Enter some text:", self)
        layout.addWidget(self.label)
        
        # Textbox
        self.textbox = QLineEdit(self)
        self.textbox.setText(default_text)
        self.textbox.selectAll()
        layout.addWidget(self.textbox)
        
        # Enter Button
        self.enter_button = QPushButton("Enter", self)
        layout.addWidget(self.enter_button)
        
        # Connect Enter Button and Enter Key to Submit
        self.enter_button.clicked.connect(self.accept)
        self.textbox.returnPressed.connect(self.enter_button.click)
    
    def get_text(self):
        if self.exec_() == QDialog.Accepted:
            return self.textbox.text()
        return None
    

if __name__ == '__main__':
    app = QApplication([])
    # image_file = util.open_image_dialog('./data/example')
    image_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/example_gt2.png"
    out_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output/test"
    window = ImageCanvas(image_file,out_dir)
    window.show()

    # window.showMaximized()
    window.raise_()
    window.activateWindow()
    app.exec()