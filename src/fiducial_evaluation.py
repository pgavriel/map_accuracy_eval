from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication
from PyQt5.QtGui import QPixmap, QTransform, QMouseEvent, QPainter, QPen, QFont, QColor, QBrush
from PyQt5.QtCore import Qt, QPointF, QSize, QPoint, QTimer
import time, os
import numpy as np
import utilities as util
import point_manip as pm
import calculate_metrics as metrics
import csv_loader

# For popup TextInputDialog
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtWidgets import QDialog

def find_midpoint(points):
    if not points:
        return None  # Handle empty list case
    
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    
    avg_x = sum_x // len(points)
    avg_y = sum_y // len(points)
    
    return (avg_x, avg_y)

def calculate_angle_and_bisector(A, B, C):
    # Convert points A, B, and C into numpy arrays
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    C = np.array(C, dtype=np.float32)
    
    # Calculate vectors AB and BC
    AB = A - B
    BC = C - B

    # Normalize vectors AB and BC
    AB_norm = AB / np.linalg.norm(AB)
    BC_norm = BC / np.linalg.norm(BC)

    # Calculate the bisector vector (normalize the sum of the unit vectors)
    bisector = AB_norm + BC_norm
    bisector /= np.linalg.norm(bisector)
    
    return bisector

def angle_between_vectors(v1, v2):
    # Calculate the angle between two vectors
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical issues
    return np.degrees(angle)

class FiducialEvaluator(QGraphicsView):
    def __init__(self, map_image_file=None,output_dir=None):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # Set Window Size
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        # Set window size relative to display size (e.g., 50% of screen width and height)
        width = int(screen_size.width() * 0.9)
        height = int(screen_size.height() * 0.7)
        self.resize(QSize(width, height))
        
        # Verify output dir
        if output_dir is None:
            default_output = "C:/Users/nullp/Projects/map_accuracy_eval/output/ua5/"
            self.output_dir = util.select_directory(default_output)
            if self.output_dir is None:
                print(f"No output directory selected, quitting.")
                exit()
        else:
            self.output_dir = output_dir
        print(f"Output Directory: {self.output_dir}")
        # Setup Log File
        self.log_name = 'fiducial_log.csv'
        self.log_file = os.path.join(self.output_dir,self.log_name)

        # Add an image
        if map_image_file is None:
            default_dir = "C:/Users/nullp/Projects/map_accuracy_eval/data/ua5/"
            map_image_file = util.open_image_dialog(default_dir)
            if map_image_file is None:
                print(f"No image selected, quitting.")
                exit()
        self.file_path = map_image_file
        print(f"Opening image file: {map_image_file}")
        pixmap = QPixmap(map_image_file)
        item = QGraphicsPixmapItem(pixmap)
        item.setOpacity(1.0)  # Set alpha
        item.setTransform(QTransform().rotate(0))  # Set rotation
        self.scene.addItem(item)
        
        # Set up zoom and pan
        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)  # Initially, no drag mode
        self.setCursor(Qt.ArrowCursor)
        self.zoom_factor = 1.15
        self.current_zoom = 1.0

        # Add data and points
        self.points = []
        self.modes = ["cylinder","splitcross"]
        self.current_mode = 0
        self.eval_mode = self.modes[self.current_mode]
        self.named_points = {}
        self.score = None
        self.midpoint = None
        self.score_reason = ""
        self.score_str = "Unscored."
        self.saved_scores = []

        self.setting_radius = False
        self.rad = 15
        self.radpoints = []

        # Set Grading Thresholds
        self.angle_tolerance = 15.0 # in Degrees
        self.distance_tolerance = 1.5 # in Fiducial Radii
        self.sc_ang_tol = 15.0 # in Degrees
        self.sc_dist_tol = 2.0 # in Fiducial Radii

        # Define some colors
        self.c1 = (255,0,0) # Neutral
        self.c2 = (0,255,0) # Good
        self.c3 = (0,0,255) # Bad

        # Add FPS Tracker
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0

        # Setup a timer to update FPS every second
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)  # Update FPS every second

        # Create control panel
        self.control_panel = None
        self.show_control_panel()

        # Set Mode
        self.switch_mode(0)
        if self.rad is not None: self.set_radius(self.rad)
     
    def wheelEvent(self, event):
        zoom_in_factor = self.zoom_factor
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.current_zoom *= zoom_factor
        print(f"[Zoom: {self.current_zoom:.2f}]")
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
                self.handle_mouse_click()
            else:
                print(f"Panned: ({delta.x()},{delta.y()})")
        super().mouseReleaseEvent(event)
        self.viewport().update()  # Trigger paintEvent when resizing   

    def paintEvent(self, event):
        self.frame_count += 1
        # First, let QGraphicsView paint the scene and everything else
        super().paintEvent(event)
        
        # Now overlay the text
        painter = QPainter(self.viewport())
        
        self.draw_on_map(painter)

        # End the painter
        painter.end()

    def draw_on_map(self, painter,saving_output=False):
        if not saving_output:
            # Draw a rectangle for the control panel background
            painter.setBrush(QBrush(QColor(0, 0, 0, 70)))
            painter.drawRect(5, 5, 250, 400)
            painter.drawRect(260, 5, 1200, 65)

            # Assemble the status text string to draw
            status_str = "{}/{} Points Chosen ({} mode)".format(len(self.points),len(self.point_names),self.eval_mode)
            if self.score is not None:
                status_str = self.score_str
            text_position = QPointF(270, 35)  # Offset from the top-left corner
            shadow_position = QPointF(274, 39)
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
            fps_position = QPointF(274, 65)
            painter.drawText(fps_position, f"FPS: {int(self.fps)}")

        # DRAW ALL SAVED EVALUATIONS
        n = 0
        for s in self.saved_scores:
            n += 1
            label = str(n)
            if s['score'] == 1:
                painter.setPen(QPen(QColor(0, 255, 0, 150),5))
                painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
            else:
                painter.setPen(QPen(QColor(255, 0, 0, 150),5))
                painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
            
            c = QPoint(s['midpoint'][0],s['midpoint'][1])
            c = self.mapFromScene(c) # Map to scene space
            r = s['rad'] * self.current_zoom
            if s['type'] == 'cylinder':
                painter.drawEllipse(c,r,r)
            elif s['type'] == 'splitcross':
                p = dict()
                for l in s['points']:
                    p[l] = self.mapFromScene(QPoint(s['points'][l][0],s['points'][l][1])) # Map to scene space
                try:
                    painter.drawLine(p['1A'],p['2C'])
                    painter.drawLine(p['1C'],p['2A'])
                    painter.drawLine(p['1A'],p['2A'])
                    painter.drawLine(p['1C'],p['2C'])
                except:
                    pass
                painter.drawEllipse(c,r,r)
            # offset = QPoint(5,-5) # Adjust position as needed
            # c += offset
            painter.setFont(QFont("Arial", 18, QFont.Bold))
            painter.drawText(c, label)

        # DRAW CURRENT POINTS
        painter.setFont(QFont("Arial", 14, QFont.Normal))
        painter.setPen(QPen(QColor(55, 55, 255, 255),5))
        scene_point = dict()
        for p in self.named_points:
            scene_pos = QPointF(self.named_points[p][0],self.named_points[p][1])
            viewport_point = self.mapFromScene(scene_pos)
            scene_point[p] = viewport_point
            x = int(viewport_point.x())
            y = int(viewport_point.y())
            label = p
            r = 2
            # Draw the point (cross with radius r)
            painter.drawLine(x,y-r,x,y+r)
            painter.drawLine(x-r,y,x+r,y)

            # Draw the label next to the point
            offset = QPoint(5,-5) # Adjust position as needed
            viewport_point += offset
            painter.drawText(viewport_point, label)

        # DRAW EVALUATION GUIDES
        if self.eval_mode == 'cylinder':
            try:
                painter.drawLine(scene_point['1A'],scene_point['1B'])
                painter.drawLine(scene_point['1A'],scene_point['2A'])
                painter.drawLine(scene_point['2A'],scene_point['2B'])
                painter.drawLine(scene_point['1B'],scene_point['2B'])
            except:
                pass
        elif self.eval_mode == 'splitcross':
            try:
                painter.drawLine(scene_point['1A'],scene_point['1B'])
                painter.drawLine(scene_point['1B'],scene_point['1C'])
                self.draw_bisector(painter,self.named_points['1A'],self.named_points['1B'],self.named_points['1C'])
                painter.drawLine(scene_point['2A'],scene_point['2B'])
                painter.drawLine(scene_point['2B'],scene_point['2C'])
                self.draw_bisector(painter,self.named_points['2A'],self.named_points['2B'],self.named_points['2C'])
            except:
                pass

        # DRAW CURRENT SCORE
        if self.score is not None:
            if self.score == 1:
                painter.setPen(QPen(QColor(0, 255, 0, 150),5))
                painter.setBrush(QBrush(QColor(0, 255, 0, 150)))
            else:
                painter.setPen(QPen(QColor(255, 0, 0, 150),5))
                painter.setBrush(QBrush(QColor(255, 0, 0, 150)))
            c = QPoint(self.midpoint[0],self.midpoint[1])
            c = self.mapFromScene(c) # Map to scene space
            r = self.rad * self.current_zoom
            painter.drawEllipse(c,r,r)

    def draw_bisector(self, painter, A, B, C, length=None):
        # Draws the angle indicator for split cross fiducials
        if length is None:
            length = self.rad * 2
        # Calculate the bisector direction vector
        bisector = calculate_angle_and_bisector(A, B, C)
        # Calculate the endpoint of the bisector line
        bisector_end = (B + bisector * length).astype(int)

        qpA = self.mapFromScene(QPoint(B[0],B[1]))
        qpB = self.mapFromScene(QPoint(bisector_end[0],bisector_end[1]))
        # Draw the bisector line
        painter.drawLine(qpA,qpB)

    def keyPressEvent(self, event):
        # Handle different key presses
        if event.key() == Qt.Key_Q: # Q: Quit
            QApplication.quit()  # Quit the application when Q is pressed

        elif event.key() == Qt.Key_F: # Fit map in window
            self.fit_in_window()
        elif event.key() == Qt.Key_I: # I: Select Image
            self.select_image()
        elif event.key() == Qt.Key_M: # M: Switch Mode
            self.switch_mode()
        elif event.key() == Qt.Key_P: # P: Set Radius
            self.set_radius()
        elif event.key() == Qt.Key_E: # E: Evaluate
            self.evaluate_points()
        elif event.key() == Qt.Key_W: # W: Store data
            self.store_data()
        elif event.key() == Qt.Key_T: # T: Remove last evaluation
            self.remove_last_data()
        elif event.key() == Qt.Key_R: # R: Reset Points
            self.reset_points()
        elif event.key() == Qt.Key_X: # X: EXPORT DATA
            self.log_stored_data()
        
        else:
            super().keyPressEvent(event)  # Call the base class method for unhandled keys

    def show_control_panel(self):
        if self.control_panel is None:
            self.control_panel = ControlPanel(self)

        # HANDLE BINDING CONTROL BUTTONS TO FUNCTIONS HERE
        
        self.control_panel.select_image_button.clicked.connect(lambda: self.select_image())
        self.control_panel.switch_mode_button.clicked.connect(lambda: self.switch_mode())
        self.control_panel.radius_button.clicked.connect(lambda: self.set_radius())
        self.control_panel.evaluate_button.clicked.connect(lambda: self.evaluate_points())
        self.control_panel.grade_next_fiducial_button.clicked.connect(lambda: self.store_data())
        self.control_panel.remove_last_fiducial_button.clicked.connect(lambda: self.remove_last_data())
        self.control_panel.reset_points_button.clicked.connect(lambda: self.reset_points())
        self.control_panel.log_data_button.clicked.connect(lambda: self.log_stored_data())

        self.control_panel.show()
           
    def update_fps(self):
        # Calculate and update FPS
        current_time = time.time()
        elapsed_time = current_time - self.last_frame_time
        self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        self.frame_count = 0
        self.last_frame_time = current_time

    def fit_in_window(self):
        # Get the size of the window (the QGraphicsView)
        self.viewport().update()
        window_size = self.size()
        print(f"[Window Size]\t{window_size.width()} x {window_size.height()}")
        # Get the size of the background image
        background_image_size = self.img_pixmap.size()  
        print(f"[Image Size]\t{background_image_size.width()} x {background_image_size.height()}")
        
        # Calculate scale factors for width and height to fit the image in the window
        scale_w = window_size.width() / background_image_size.width()
        scale_h = window_size.height() / background_image_size.height()
        
        # Use the smaller scale factor to maintain aspect ratio
        scale_factor = min(scale_w, scale_h)
        print(f"[Scales] \t{scale_w:.2f} x {scale_h:.2f} (Scaling by {scale_factor:.2f})")
        
        # Apply scaling to the QGraphicsView
        self.resetTransform()  # Reset any existing transformations
        self.scale(scale_factor, scale_factor)
        self.current_zoom = scale_factor
        print(f"[Zoom: {self.current_zoom:.2f}]")

        # Center the image in the view
        self.centerOn(background_image_size.width() / 2, background_image_size.height() / 2)
        self.viewport().update()

    def save_screenshot(self,save_file):
        # Fit image in window
        self.fit_in_window()
        # Create a painter to render the view onto the QPixmap
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        # Render the QGraphicsView onto the pixmap
        self.render(painter)
        # Add custom draw procedures
        self.draw_on_map(painter,True)
        # End the painter
        painter.end()
        # Save the pixmap to the chosen file
        pixmap.save(save_file)
        print(f"Saved Image: {save_file}")

    def select_image(self, map_image_file=None):
        # Acquire image file path
        if map_image_file is None:
            map_image_file = util.open_image_dialog('./data/example')
            if map_image_file is None:
                print(f"No image selected, quitting.")
                QApplication.quit()
        self.file_path = map_image_file

        # Attempt to load image
        print(f"Opening image file: {map_image_file}")
        pixmap = QPixmap(map_image_file)
        item = QGraphicsPixmapItem(pixmap)
        item.setOpacity(1.0)  # Set alpha
        item.setTransform(QTransform().rotate(0))  # Set rotation

        # Clear the scene, add new image, refresh image
        self.scene.clear()
        self.scene.addItem(item)
        self.resetTransform()
        self.viewport().update()

        # Reset any saved scores or points
        self.saved_scores = []
        self.reset_points()

    def set_radius(self,r=None):
        if r is None:
            print(f"Entering mode to set fiducial radius...\nPlease select 2 points: ")
            self.radpoints = []
            self.setting_radius = True
        else:
            self.rad = r
            self.control_panel.radius_label.setText(f"Radius: {self.rad}")
            print(f"Radius set to {r}")

    def handle_mouse_click(self):
        x = int(self.mouse_scene_pos.x())
        y = int(self.mouse_scene_pos.y())
        
        # If setting radius...
        if self.setting_radius:
            self.radpoints.append((x,y))
            print(f"Radius point {len(self.radpoints)} selected: ({x},{y})")
            if len(self.radpoints) >= 2:
                self.rad = pm.calc_distance(self.radpoints[0],self.radpoints[1])
                if self.current_mode == 0: # cylinder
                    self.rad = self.rad / 2 
                self.rad = int(self.rad)
                print(f"Radius set to {self.rad}.")
                self.control_panel.radius_label.setText(f"Radius: {self.rad}")
                self.setting_radius = False
            return
            
        # Otherwise, try to add point for evaluation...   
        if len(self.points) < len(self.point_names):
            point_name = self.point_names[len(self.points)]
            self.points.append((x, y))
            self.named_points[point_name] = (x, y)
            # Automatically evaluate if all points selected
            if len(self.points) == len(self.point_names):
                self.evaluate_points()
        else:
            print("All points have been selected.")

    def switch_mode(self,to_mode=None):
        if to_mode is None:
            self.current_mode = (self.current_mode + 1) % len(self.modes)
        else:
            self.current_mode = to_mode % len(self.modes)
        self.eval_mode = self.modes[self.current_mode]

        if self.current_mode == 0: # "cylinder":
            self.point_names = ["1A", "1B", "2A", "2B"]
        elif self.current_mode == 1: # "splitcross":
            self.point_names = ["1A","1B","1C","2A","2B","2C"]

        # Clear all current points and saved evals
        #TODO: Figure out how both types can be evaluated simultaneously
        self.reset_points()
        self.saved_scores = []

        self.control_panel.mode_label.setText(f"Current Mode: {self.eval_mode}")
        print(f"Current Mode: {self.current_mode} - {self.eval_mode}")

        # Refresh Image
        self.viewport().update()

    def reset_points(self):
        self.score_str = "Unscored."
        self.score = None
        self.points = []
        self.named_points = {}
        # Refresh Image
        self.viewport().update()

    def remove_last_data(self):
        if len(self.saved_scores) > 0:
            s = self.saved_scores.pop()
            print(f"Removed Data: {s}")
        else:
            print(f"Nothing removed.")
        self.reset_points()

    def store_data(self):
        # Store the points and score, e.g., append to a list or save to a file
        print(f"Storing points: {self.named_points} with score: {self.score}")
        fiducial_data = dict()
        fiducial_data['type'] = self.eval_mode
        fiducial_data['score'] = self.score
        fiducial_data['reason'] = self.score_reason
        fiducial_data['midpoint'] = self.midpoint
        fiducial_data['rad'] = self.rad
        fiducial_data['points'] = self.named_points
        self.saved_scores.append(fiducial_data)
        print(f"STORED: {fiducial_data}")
        self.reset_points()

    def evaluate_points(self):
        if len(self.points) > 0: 
            if self.rad is None:
                print("Must set Radius first.")
                return
            if self.score is not None:
                print("Score already calculated")
                return
            if self.current_mode == 0: # "cylinder":
                print("Evaluating Cylinder...")
                self.calculate_cylinder_score()
            elif self.current_mode == 1: # splitcross
                print("Evaluating Splitscross...")
                self.calculate_cross_score()
            print(f"Score: {self.score}")
            # Draw additional data on the image, such as lines connecting the points
            self.viewport().update()
        else:
            print("Must select at least 1 point.")

    def next_fiducial(self):
        if self.score is None:
            print("Must calculate score first")
            return
        self.store_data()
        print("Stored points for current fiducial. Ready for next fiducial.")
        self.reset_points()

    def calculate_cylinder_score(self):
        self.score_reason = ""
        self.score = 0
        self.midpoint = find_midpoint(self.points)

        # TEST COMPLETE 
        if len(self.points) != len(self.point_names):
            self.score_reason = "[INCOMPLETE]"
            print(self.score_reason)
            self.score_str = f"Cylinder FAILED: {self.score_reason}"
            return
        
        # TEST DISTANCE
        f1_diam = pm.calc_distance(self.named_points['1A'],self.named_points['1B'])
        f2_diam = pm.calc_distance(self.named_points['2A'],self.named_points['2B'])
        avg_diam = (f1_diam + f2_diam) /2
        dist_tolerance = avg_diam * self.distance_tolerance
        sidea_dist = pm.calc_distance(self.named_points['1A'],self.named_points['2A'])
        sideb_dist = pm.calc_distance(self.named_points['1B'],self.named_points['2B'])
        
        dist_pass = False
        if sidea_dist <= dist_tolerance and sideb_dist <= dist_tolerance:
            dist_pass = True
        else: 
            self.score_reason = self.score_reason + f"[DIST > {self.distance_tolerance:.1f}R]"
        print(f"[Diameters Found] F1:{f1_diam:.1f}  F2:{f2_diam:.1f}  AVG:{avg_diam:.1f}")
        print(f"[Distances] SideA:{sidea_dist:.2f}  SideB:{sideb_dist:.2f} Thresh:{dist_tolerance:.2f} PASS?:{dist_pass}")
        
        # TEST ANGLE
        f1_ang = pm.calc_angle(self.named_points['1A'],self.named_points['1B'])
        if f1_ang < 0: f1_ang += 360.0
        f2_ang = pm.calc_angle(self.named_points['2A'],self.named_points['2B'])
        if f2_ang < 0: f2_ang += 360.0
        diff_ang = abs(f1_ang - f2_ang)
        diff_ang = min(diff_ang, 360 - diff_ang)
        angle_pass = False
        if diff_ang <= self.angle_tolerance:
            angle_pass = True
        else:
            self.score_reason = self.score_reason + f"[ANGLE > {self.angle_tolerance:.1f}DEG]"
        print(f"[Angles Found] F1:{f1_ang:.1f}  F2:{f2_ang:.1f}  AbsDiff:{diff_ang:.1f} PASS?:{angle_pass}")
        
        # See if Fiducial points pass both tests and score accordingly
        if dist_pass and angle_pass:
            self.score = 1
            print("[PASSED]")
            c = self.c2
            self.score_str = f"Cylinder PASSED: [D1:{sidea_dist:.1f}][D2:{sideb_dist:.1f}][ANG:{diff_ang:.1f}deg]"
        else:
            print("[FAILED]")
            print(self.score_reason)
            self.score_str = f"Cylinder FAILED: {self.score_reason}"

    def calculate_cross_score(self):
        self.score = 0
        self.score_reason = ""
        self.midpoint = find_midpoint(self.points)

        # TEST COMPLETE 
        if len(self.points) != len(self.point_names):
            self.score_reason = "[INCOMPLETE]"
            print(self.score_reason)
            self.score_str = f"Splitcross FAILED: {self.score_reason}"
            return
        
        # TEST ANGLE
        angle_pass = False
        # Calculate bisector vectors for two sets of points
        bisector1 = calculate_angle_and_bisector(self.named_points["1A"], self.named_points["1B"], self.named_points["1C"])
        bisector2 = calculate_angle_and_bisector(self.named_points["2A"], self.named_points["2B"], self.named_points["2C"])
        # Find the angle between the two bisectors
        angle = angle_between_vectors(bisector1, bisector2)
        # Check if the angle is within the range (180 ± tolerance)
        low_tol = 180 - self.sc_ang_tol
        high_tol = 180 + self.sc_ang_tol
        if low_tol <= angle <= high_tol: 
            angle_pass = True
        else:
            self.score_reason = self.score_reason + f"[ANGLE > {self.sc_ang_tol:.1f}DEG]"
        print(f"[Angles Found] Diff:{angle:.1f}  Tolerance:{self.sc_ang_tol:.1f} ({low_tol}-{high_tol}) PASS?:{angle_pass}")
        
        # TEST DISTANCE
        dist_pass = False
        center_dist = pm.calc_distance(self.named_points['1B'],self.named_points['2B'])
        if center_dist <= (self.rad * self.sc_dist_tol):
            dist_pass = True
        else:
            self.score_reason = self.score_reason + f"[DIST > {self.sc_dist_tol:.1f}R]"
        print(f"[Distances] Radius:{self.rad:.2f}  Distance:{center_dist:.2f} Tolerance:{self.rad * self.sc_dist_tol:.2f} PASS?:{dist_pass}")


        if dist_pass and angle_pass:
            self.score = 1
            print("[PASSED]")
            self.score_str = f"Splitcross PASSED: [DIST:{center_dist:.1f}][ANG:{angle:.1f}deg]"
            c = self.c2
        else:
            print("[FAILED]")
            print(self.score_reason)
            self.score_str = f"Splitcross FAILED: {self.score_reason}"

    def log_stored_data(self,verbose=True):
        headers = ['Timestamp','Image File','Evaluation Type',
                   'Fiducials Passed','Fiducials Scored','Score','Reasons']
        passed = 0
        count = 0
        reasons = dict()
        for f in self.saved_scores:
            count += 1
            if f['score'] == 1: passed += 1
            else:
                reasons[count] = f['reason']

        # Rescale
        # scale_factor = 1/self.current_zoom
        # self.current_zoom *= scale_factor
        # self.scale(scale_factor,scale_factor)
        # print(f"NEW ZOOM {self.current_zoom}")

        ts = util.timestamp()
        image_name = ts + "_fiducials.png"
        image_file = os.path.join(self.output_dir,image_name)
        log_list = [ts,os.path.basename(self.file_path),self.eval_mode,
                    passed,len(self.saved_scores),passed/len(self.saved_scores),reasons]
        metrics.log_to_csv(self.log_file, log_list,header=headers,verbose=verbose)
        self.save_screenshot(image_file)
        print(f"Image {image_file} saved.")

class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Control Panel")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        
        # self.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        # Set the window flags to make it a separate window but still linked to the parent
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setGeometry(5, 5, 250, 400)
        # Create layout for the control panel
        layout = QVBoxLayout()

        # Create and add buttons and labels to the layout
        self.select_image_button = QPushButton("Select Image (I)")
        self.switch_mode_button = QPushButton("Switch Mode (M)")
        self.mode_label = QLabel("Current Mode: ")
        self.radius_label = QLabel("Radius: ")
        self.radius_button = QPushButton("Set Radius (P)")
        self.evaluate_button = QPushButton("Evaluate (E)")
        self.grade_next_fiducial_button = QPushButton("Next Fiducial (W)")
        self.remove_last_fiducial_button = QPushButton("Remove Last Fiducial (T)")
        self.reset_points_button = QPushButton("Reset Points (R)")
        self.log_data_button = QPushButton("Log Data (X)")
        self.quit_button = QPushButton("Quit (Q)")

        # Add widgets to the layout
        layout.addWidget(self.select_image_button)

        # mode_layout = QHBoxLayout()
        layout.addWidget(self.switch_mode_button)
        layout.addWidget(self.mode_label)
        # layout.addLayout(mode_layout)
        
        # radius_layout = QHBoxLayout()
        layout.addWidget(self.radius_button)
        layout.addWidget(self.radius_label)
        # layout.addLayout(radius_layout)
        
        layout.addWidget(self.evaluate_button)
        layout.addWidget(self.grade_next_fiducial_button)
        layout.addWidget(self.remove_last_fiducial_button)
        layout.addWidget(self.reset_points_button)
        layout.addWidget(self.log_data_button)
        layout.addWidget(self.quit_button)

        # Set the layout for the control panel window
        self.setLayout(layout)

        # Connect the buttons to actions
        self.quit_button.clicked.connect(self.quit_application)

    def quit_application(self):
        # Handle quit action
        self.close()
        self.parent().close()  # Optionally close the main window as well

if __name__ == '__main__':
    app = QApplication([])
    # image_file = util.open_image_dialog('./data/example')
    image_file = None
    # image_file = "C:/Users/nullp/Projects/map_accuracy_eval/data/example/example_gt2.png"
    out_dir = None
    # out_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output/test"
    window = FiducialEvaluator(image_file,out_dir)
    window.show()

    # window.showMaximized()
    window.raise_()
    window.activateWindow()
    app.exec()