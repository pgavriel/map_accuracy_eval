import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
import point_manip as pm
import calculate_metrics as metrics
import utilities as util

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

def draw_bisector(image, A, B, C, length=100, color=(0, 255, 0), thickness=2):
    # Calculate the bisector direction vector
    bisector = calculate_angle_and_bisector(A, B, C)
    # Calculate the endpoint of the bisector line
    bisector_end = (B + bisector * length).astype(int)
    # Draw the bisector line
    cv2.line(image, tuple(B), tuple(bisector_end), (0, 0, 255), thickness)

    return image

def angle_between_vectors(v1, v2):
    # Calculate the angle between two vectors
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical issues
    return np.degrees(angle)

class FiducialEvaluator:
    def __init__(self):
        self.output_dir = 'C:/Users/nullp/Projects/map_accuracy_eval/output/fiducial_method'
        self.log_name = 'fiducial_log.csv'
        self.log_file = os.path.join(self.output_dir,self.log_name)
        self.points = []
        self.modes = ["cylinder","splitcross"]
        self.current_mode = 0
        self.eval_mode = self.modes[self.current_mode]
        self.named_points = {}
        self.image = None
        self.display_image = None
        self.angle_tolerance = 15.0 # in Degrees
        self.distance_tolerance = 1.0 # in Fiducial Radii
        self.sc_ang_tol = 15.0 # in Degrees
        self.sc_dist_tol = 1.0 # in Fiducial Radii
        self.score = None
        self.score_reason = ""
        self.saved_scores = []

        # Create OpenCV window and set mouse callback
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_callback)
        self.select_image()

        # Initialize Tkinter root
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

        # Define some colors
        self.c1 = (255,0,0) # Neutral
        self.c2 = (0,255,0) # Good
        self.c3 = (0,0,255) # Bad

        # Create control panel
        self.construct_control_panel()

        # Set Mode
        self.switch_mode(0)

        # Start the periodic OpenCV event processing
        self.root.after(10, self.process_opencv_events)

    def construct_control_panel(self):
        self.control_panel = tk.Toplevel()
        self.control_panel.title("Control Panel")

        load_button = tk.Button(self.control_panel, text="Select Image", command=self.select_image)
        load_button.pack(padx=10, pady=10)

        self.mode_lbl = tk.Label(self.control_panel,text=f"Current Mode: {self.eval_mode}")
        self.mode_lbl.pack(padx=10, pady=2)
        mode_button = tk.Button(self.control_panel, text="Switch Mode", command=self.switch_mode)
        mode_button.pack(padx=10, pady=2)

        evaluate_button = tk.Button(self.control_panel, text="Evaluate (E)", command=self.evaluate_points)
        evaluate_button.pack(padx=10, pady=10)

        next_fiducial_button = tk.Button(self.control_panel, text="Next Fiducial (N)", command=self.next_fiducial)
        next_fiducial_button.pack(padx=10, pady=10)

        rem_fiducial_button = tk.Button(self.control_panel, text="Remove Last Ficucial (T)", command=self.remove_last_data)
        rem_fiducial_button.pack(padx=10, pady=10)

        reset_button = tk.Button(self.control_panel, text="Reset Points (R)", command=self.reset_points)
        reset_button.pack(padx=10, pady=10)

        log_button = tk.Button(self.control_panel, text="Log Data (X)", command=self.log_stored_data)
        log_button.pack(padx=10, pady=10)

        quit_button = tk.Button(self.control_panel, text="Quit (Q)", command=self.quit_app)
        quit_button.pack(padx=10, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.file_path = file_path
                self.display_image = self.image.copy()
                self.draw_stored_data()
                cv2.imshow("Image", self.display_image)
                self.points = []
                self.named_points = {}

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < len(self.point_names):
                point_name = self.point_names[len(self.points)]
                self.points.append((x, y))
                self.named_points[point_name] = (x, y)
                cv2.circle(self.display_image, (x, y), 3, self.c1, -1)
                cv2.putText(self.display_image, point_name, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.c1, 1)
                
                cv2.imshow("Image", self.display_image)
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

        self.mode_lbl.config(text=f"Current Mode: {self.eval_mode}")
        print(f"Current Mode: {self.current_mode} - {self.eval_mode}")

        self.display_image = self.image.copy()
        cv2.imshow("Image", self.display_image)

    def evaluate_points(self):
        if len(self.points) > 0: 
            if self.current_mode == 0: # "cylinder":
                self.calculate_cylinder_score()
                print(f"Score: {self.score}")
            elif self.current_mode == 1: # splitcross
                self.calculate_cross_score()
                print("Eval splitscross...")
            # Draw additional data on the image, such as lines connecting the points
            self.draw_evaluation()
            cv2.imshow("Image", self.display_image)
        else:
            print("Must select at least 1 point.")

    def next_fiducial(self):
        if self.score is None:
            print("Must calculate score first")
            return
        # Store the points and score before moving to the next fiducial
        # if len(self.points) == len(self.point_names):
            
        self.store_data()
        print("Stored points for current fiducial. Ready for next fiducial.")
        self.reset_points()

    # TODO: Make cylinder specific, make another function for angle scoring
    def calculate_cylinder_score(self):
        if self.score is not None:
            print("Score already calculated")
            return
        
        self.score_reason = ""
        self.score = 0

        # TEST COMPLETE 
        if len(self.points) != len(self.point_names):
            self.score_reason = "[INCOMPLETE]"
            self.midpoint = find_midpoint(self.points)
            self.rad = 15
            c = self.c3
            cv2.circle(self.display_image, self.midpoint, self.rad, c, 2)
            print(self.score_reason)
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
        else:
            print("[FAILED]")
            c = self.c3
            print(self.score_reason)
        self.midpoint = find_midpoint(self.points)
        # print(self.midpoint)
        self.rad = int(avg_diam/2)
        cv2.circle(self.display_image, self.midpoint, int(avg_diam/2), c, 2)

    def calculate_cross_score(self):
        self.score = 0
        self.score_reason = ""

        # TEST COMPLETE 
        if len(self.points) != len(self.point_names):
            self.score_reason = "[INCOMPLETE]"
            self.midpoint = find_midpoint(self.points)
            self.rad = 15
            c = self.c3
            self.draw_cross(self.midpoint,self.rad,c)
            print(self.score_reason)
            return
        
        # Find Fiducial Radius
        dist_sum = pm.calc_distance(self.named_points['1A'],self.named_points['1B']) \
                 + pm.calc_distance(self.named_points['1C'],self.named_points['1B']) \
                 + pm.calc_distance(self.named_points['2A'],self.named_points['2B']) \
                 + pm.calc_distance(self.named_points['2C'],self.named_points['2B'])
        self.rad = dist_sum // 4
        draw_bisector(self.display_image,self.named_points["1A"],self.named_points["1B"],self.named_points["1C"],length=self.rad*2)
        draw_bisector(self.display_image,self.named_points["2A"],self.named_points["2B"],self.named_points["2C"],length=self.rad*2)


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
            c = self.c2
        else:
            print("[FAILED]")
            c = self.c3
            print(self.score_reason)
        self.midpoint = find_midpoint(self.points)

    def draw_cross(self,midpoint,rad,color):
        rad = int(rad)
        print(f"mp: {midpoint}, rad: {rad}")
        p1 = (midpoint[0]-rad,midpoint[1]-rad)
        print(p1)
        p2 = (midpoint[0]+rad,midpoint[1]+rad)
        cv2.line(self.display_image, p1, p2, color, 2)
        p1 = (midpoint[0]+rad,midpoint[1]-rad)
        p2 = (midpoint[0]-rad,midpoint[1]+rad)
        cv2.line(self.display_image, p1, p2, color, 2)

    def store_data(self):
        # self.score = None
        # Store the points and score, e.g., append to a list or save to a file
        print(f"Storing points: {self.named_points} with score: {self.score}")
        fiducial_data = dict()
        fiducial_data['type'] = self.eval_mode
        fiducial_data['score'] = self.score
        fiducial_data['reason'] = self.score_reason
        fiducial_data['midpoint'] = self.midpoint
        fiducial_data['rad'] = self.rad
        self.saved_scores.append(fiducial_data)
        print(f"STORED: {fiducial_data}")
        self.reset_points()

    def draw_stored_data(self):
        d = 1
        for f in self.saved_scores:
            if f['score'] == 1:
                c = self.c2
            else:
                c = self.c3
            if self.current_mode == 0: # cylinder
                cv2.circle(self.display_image, f['midpoint'],f['rad'], c, 2)
            elif self.current_mode == 1: # cross
                self.draw_cross(f['midpoint'],f['rad'],c)
            cv2.putText(self.display_image, str(d), f['midpoint'], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(self.display_image, str(d), f['midpoint'], cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 1)
            d += 1

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

        ts = util.timestamp()
        image_name = ts + "_fiducials.png"
        image_file = os.path.join(self.output_dir,image_name)
        log_list = [ts,os.path.basename(self.file_path),self.eval_mode,
                    passed,len(self.saved_scores),passed/len(self.saved_scores),reasons]
        metrics.log_to_csv(self.log_file, log_list,header=headers,verbose=verbose)
        cv2.imwrite(image_file,self.display_image)
        print(f"Image {image_file} saved.")

    def draw_evaluation(self):
        if self.current_mode == 0: # cylinder
            # Draw lines connecting points, etc.
            points = list(self.named_points.values())
            for i in range(len(points) - 1):
                cv2.line(self.display_image, points[i], points[i + 1], self.c1, 1)
            # Example: Draw a line connecting the first and last points
            if len(points) > 2:
                cv2.line(self.display_image, points[0], points[-1], self.c1, 1) 
        elif self.current_mode == 1: # splitcross
            if len(self.named_points) >= 3:
                cv2.line(self.display_image, self.named_points["1A"], self.named_points["1B"], self.c1, 2)
                cv2.line(self.display_image, self.named_points["1C"], self.named_points["1B"], self.c1, 2)
            if len(self.named_points) == len(self.point_names):
                cv2.line(self.display_image, self.named_points["2A"], self.named_points["2B"], self.c1, 2)
                cv2.line(self.display_image, self.named_points["2C"], self.named_points["2B"], self.c1, 2)

    def remove_last_data(self):
        if len(self.saved_scores) > 0:
            s = self.saved_scores.pop()
            print(f"Removed Data: {s}")
        self.reset_points()

    def reset_points(self):
        self.score = None
        self.points = []
        self.named_points = {}
        if self.image is not None:
            self.display_image = self.image.copy()
            self.draw_stored_data()
            cv2.imshow("Image", self.display_image)

    def process_opencv_events(self):
        # Handle OpenCV window events and keypresses
        if self.display_image is not None:
            cv2.imshow("Image", self.display_image)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('e'):  # Evaluate points (Key: 'e')
            self.evaluate_points()
        elif key == ord('r'):  # Reset points (Key: 'r')
            self.reset_points()
        elif key == ord('n'):  # Next Fiducial (Key: 'n')
            self.next_fiducial()
        elif key == ord('t'): # Remove Last Fiducial (Key: 't')
            self.remove_last_data()
        elif key == ord('q'):  # Quit (Key: 'q')
            self.quit_app()
        elif key == ord('x'):  # Log Data (Key: 'x')
            self.log_stored_data()

        # Schedule the next call to this function (e.g., 10ms later)
        self.root.after(10, self.process_opencv_events)

    def quit_app(self):
        self.root.quit()
        cv2.destroyAllWindows()

    def run(self):
        # Start the Tkinter main loop
        self.root.mainloop()

# Run the application
if __name__ == "__main__":
    app = FiducialEvaluator()
    app.run()
