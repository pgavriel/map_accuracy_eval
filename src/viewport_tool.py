import cv2
# import utils as ut
import numpy as np

def scale_and_fill(image, desired_width, desired_height):
    # Calculate the aspect ratio of the input image
    aspect_ratio = image.shape[1] / image.shape[0]

    # Calculate the scaling factors for width and height
    scale_factor_width = desired_width / image.shape[1]
    scale_factor_height = desired_height / image.shape[0]

    # Choose the smaller scaling factor to maintain aspect ratio
    scale_factor = min(scale_factor_width, scale_factor_height)

    # Calculate the new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a black canvas of the desired dimensions
    output_image = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

    # Calculate the position to paste the resized image
    x_offset = (desired_width - new_width) // 2
    y_offset = (desired_height - new_height) // 2

    # Paste the resized image onto the canvas
    output_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return output_image

def fix_dimension(image, fixed_size=1000, dim="height",verbose=True):
    original_aspect = image.shape[1] / image.shape[0]
    if verbose: print(f"Original dim: {image.shape[1]}x{image.shape[0]} aspect: {original_aspect}")
    scale = 1
    if dim =="height":
        new_height = int(fixed_size)
        new_width = int(new_height * original_aspect)
        scale = new_height / image.shape[0]
    elif dim == "width":
        new_width = int(fixed_size)
        new_height = int(new_width / original_aspect)
        scale = new_width / image.shape[0]
    else:
        print("ERROR: Set dim to \"width\" or \"height\"")
        return image, -1
    
    resized_image = cv2.resize(image, (new_width, new_height))
    if verbose: 
        new_aspect = resized_image.shape[1] / resized_image.shape[0]
        print(f"New dim: {resized_image.shape[1]}x{resized_image.shape[0]} aspect: {new_aspect}  scale: {scale}")
    return resized_image, scale

def mouse_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP:
        print(f"Mouse Cick: {x},{y}")
        pass
        
    return

class Viewport:
    def __init__(self, image, x=300, y=300, w=300, h=300, a=0):
        self.image = image
        self.view = None
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.a = a
        # Rotated height/width
        self.rw = int(w * np.abs(np.cos(np.radians(a))) + h * np.abs(np.sin(np.radians(a))))
        self.rh = int(w * np.abs(np.sin(np.radians(a))) + h * np.abs(np.cos(np.radians(a))))
        # self.scale = 1.0

        #Movement
        self.dx = 0
        self.dy = 0
        self.da = 0

        self.debug = True

    def stop(self):
        self.dx = 0
        self.dy = 0
        self.da = 0

    def reset(self):
        self.stop()
        self.x = self.image.shape[1]//2
        self.y = self.image.shape[0]//2
        self.w = self.image.shape[1]
        self.h = self.image.shape[0]
        self.a = 0
        self.rw = int(self.w * np.abs(np.cos(np.radians(self.a))) + self.h * np.abs(np.sin(np.radians(self.a))))
        self.rh = int(self.w * np.abs(np.sin(np.radians(self.a))) + self.h * np.abs(np.cos(np.radians(self.a))))

    def get_state(self):
        return [self.x, self.y, self.w, self.h, self.a]

    def set_state(self,state):
        '''
        State should be format [x,y,w,h,a]
        '''
        if len(state) != 5:
            print("Invalid state, not set. ", state)
            return

        self.x = int(state[0])
        self.y = int(state[1])
        self.w = int(state[2])
        self.h = int(state[3])
        self.a = state[4]
        # if self.debug:
        #     print("State set.")

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.a = (self.a + self.da) % 360
        self.check_bounds()

        # Create a copy of the original image
        display_image = self.image.copy()

        # Calculate rotated box region dimensions
        self.rw = int(self.w * np.abs(np.cos(np.radians(self.a))) + self.h * np.abs(np.sin(np.radians(self.a))))
        self.rh = int(self.w * np.abs(np.sin(np.radians(self.a))) + self.h * np.abs(np.cos(np.radians(self.a))))

        # Extract the region within the rotated box
        rotated_box_region = cv2.getRectSubPix(display_image, (self.rw, self.rh), (self.x , self.y ))

        # Unrotate the viewport in the extracted region
        if abs(np.sin(np.radians(self.a))) > abs(np.cos(np.radians(self.a))):
            M = cv2.getRotationMatrix2D((self.rw / 2, self.rh / 2), self.a+90, 1)
        else:
            M = cv2.getRotationMatrix2D((self.rw / 2, self.rh / 2), self.a, 1)
        unrotated_box_region = cv2.warpAffine(rotated_box_region, M, (self.rw, self.rh))

        # Extract the viewport from the unrotated image
        if abs(np.sin(np.radians(self.a))) > abs(np.cos(np.radians(self.a))):
            viewport_region = cv2.getRectSubPix(unrotated_box_region, (self.h, self.w), (self.rw/2 , self.rh/2 ))
            # Transpose the image (swap width and height)
            viewport_region = cv2.transpose(viewport_region)
            # Flip the transposed image horizontally
            viewport_region = cv2.flip(viewport_region, 1)
        else:
            viewport_region = cv2.getRectSubPix(unrotated_box_region, (self.w, self.h), (self.rw/2 , self.rh/2 ))

        self.view = viewport_region

        if self.debug:
            # Draw Bounding Box
            box = cv2.boxPoints(((self.x, self.y), (self.rw, self.rh), 0))
            box = np.intp(box)
            cv2.drawContours(display_image, [box], 0, (0, 0, 255), 2)

            # Draw the viewport box
            box = cv2.boxPoints(((self.x, self.y), (self.w, self.h), self.a))
            box = np.intp(box)
            cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)

            # Draw Direction Indicator
            try:
                line_length = self.h/2
                line_endpoint = (int(self.x  + line_length * np.sin(np.radians(self.a))),
                                int(self.y  - line_length * np.cos(np.radians(self.a))))
                cv2.line(display_image, (self.x , self.y ),
                        line_endpoint, (255, 0, 0), 2)
            except:
                print("oops",(self.x , self.y ), line_endpoint)

            # Show the display image with the viewport
            # cv2.imshow('Viewport Controls', display_image)
            # Show the extracted and unrotated region
            # cv2.imshow('Rotated Region', rotated_box_region)
            # Show the corrected extracted region
            # cv2.imshow('Adjusted Region', unrotated_box_region)
            # Show the viewport
            # cv2.imshow('Viewport Region', viewport_region)


    def move(self, direction, step=10, mode="absolute"):
        '''
        Move viewport
        direction = ["up","left","right","down"]
        step = int distance
        mode = ["absolute","relative"]
        Absolute moves with respect to parent image
        Relative moves with respect to viewport accounting for rotation
        '''
        if mode == "absolute":
            if direction == "up":
                self.y = max(self.rh/2, self.y - step)
            elif direction == "left":
                self.x = max(self.rw/2, self.x - step)
            elif direction == "down":
                self.y = min(self.image.shape[0] - (self.rh/2), self.y + step)
            elif direction == "right":
                self.x = min(self.image.shape[1] - (self.rw/2), self.x + step)
            else:
                print("[ABS] Unknown direction: [up, down, left, right], got ",direction)
        elif mode == "relative":
            #Treat directions as a vector with respect to the current
            if direction == "up":
                self.x -= int(step * np.sin(np.radians(-self.a)))
                self.y -= int(step * np.cos(np.radians(-self.a)))
            elif direction == "left":
                self.x -= int(step * np.cos(np.radians(-self.a)))
                self.y += int(step * np.sin(np.radians(-self.a)))
            elif direction == "down":
                self.x += int(step * np.sin(np.radians(-self.a)))
                self.y += int(step * np.cos(np.radians(-self.a)))
            elif direction == "right":
                self.x += int(step * np.cos(np.radians(-self.a)))
                self.y -= int(step * np.sin(np.radians(-self.a)))
            else:
                print("[REL] Unknown direction: [up, down, left, right], got ",direction)
        else:
            print("Unknown mode: [absolute, relative], got ", mode)

        # self.check_bounds()


    def check_bounds(self):
        # Right Bound Check
        if self.x > self.image.shape[1] - (self.rw/2):
            # print("r:",self.x - (self.image.shape[1] - (self.rw//2)))
            self.x = (self.image.shape[1] - (self.rw//2))
        # Bottom Bound Check
        if self.y > self.image.shape[0] - (self.rh/2):
            self.y = (self.image.shape[0] - (self.rh//2))
            # print("d:",self.y - (self.image.shape[0] - (self.rh//2)))
        # Left Bound Check
        if self.x < self.rw/2: # Check Left Bound
            self.x = self.rw//2
            # print("l:",self.x - (self.rw/2))
        # Top Bound Check
        if self.y < self.rh/2: # Check Top Bound
            # print("u:",self.y - (self.rh/2) )
            self.y = self.rh//2

class ViewportAnimator:
    def __init__(self):
        self.playing = False

        self.states = []
        self.steps = []

        self.current_step = 0
        self.current_state = []

        self.modes = ["jump","interpolate"]
        self.mode = 1
        self.debug = True

    def __str__(self):
        s = "ANIMATOR INFO:\n"
        if len(self.states) == 0:
            s += "Empty."
        else:
            s += "Current State:{}   \tStep:{}\n".format(self.current_state,self.current_step)
            for i in range(len(self.states)):
                s += " {}- State:{}   \tSteps:{}\n".format(i+1,self.states[i],self.steps[i])
        return s

    def update(self):
        if len(self.states) == 0:
            return

        if self.playing:
            # Animate
            # Determine state for current step

            state_a = 0
            step = self.current_step
            while step > self.steps[state_a]:
                step -= self.steps[state_a]
                state_a += 1
            state_b = (state_a + 1) % len(self.states)
            if self.debug:
                print("State A[{}]: {}\t State B[{}]: {}".format(state_a,self.states[state_a],state_b,self.states[state_b]))

            if self.mode == 0 : # Jump
                self.current_state = self.states[state_a]
            elif self.mode == 1: # Interpolate
                state_delta = []
                for i in range(4):
                    state_delta.append(self.states[state_b][i]-self.states[state_a][i])
                # Find smallest angle delta
                abs_diff = abs(self.states[state_b][4] - self.states[state_a][4])

                if abs_diff > 180:
                    direction = -1 if self.states[state_b][4] > self.states[state_a][4] else 1
                else:
                    direction = -1 if self.states[state_a][4] > self.states[state_b][4] else 1
                print("A:{} B:{} ABS:{} DIR:{}".format(self.states[state_a][4],self.states[state_b][4],abs_diff,direction))
                if abs_diff > 180:
                    state_delta.append(direction * (360 - abs_diff))
                else:
                    state_delta.append(direction * abs_diff)

                print("State Delta: {}".format(state_delta))
                for i in range(5):
                    state_delta[i] = (state_delta[i] / self.steps[state_b]) * (step)
                print("State Delta2: {}, step: {}".format(state_delta,(step)))
                for i in range(5):
                    state_delta[i] = state_delta[i] + self.states[state_a][i]
                self.current_state = state_delta

            # Update current step, add behavior options [loop, reverse, random]
            self.current_step = (self.current_step + 1) % sum(self.steps)
            if self.debug:
                print("Current step: " ,self.current_step)
        else:
            # Animator paused
            pass

    def add_state(self,state,steps=40):
        '''
        State should be format [x,y,w,h,a]
        '''
        if len(state) != 5:
            print("Invalid state length, not added.")
            return

        self.states.append(state)
        self.steps.append(steps)
        print("Added State {}, for {} steps".format(state,steps))


    def playpause(self):
        self.playing = not self.playing

    def reset(self):
        self.playing = False
        self.position = 0
        self.states = []
        self.steps = []

    def print_state(self):
        pass
        # implement to print a copy pasteable code snippet to

if __name__ == "__main__":
    # Load an image
    im_file = "C:/Users/nullp/Projects/map_accuracy_eval/input/mapbkg_skydio.jpeg"
    # im_file = "C:/Users/nullp/Projects/map_accuracy_eval/input/420-rough-raster.png"
    image = cv2.imread(im_file)  # Replace with your image path
    if image is None:
        print("Error loading image.")
        exit()
    cv2.namedWindow("Viewport Region")
    cv2.setMouseCallback("Viewport Region", mouse_event)
    vp = Viewport(image,w=550)
    vp_max = 2500
    vp.reset()
    h = image.shape[0]
    w = image.shape[1]
    if max(h,w) > vp_max:
        pass
        if h >= w:
            new_state = [0, 0, int(w*(vp_max/h)),vp_max,0]
        else:
            new_state = [0, 0, vp_max, int(h*(vp_max/w)), 0]
        print(f"New State: {new_state}")
        vp.set_state(new_state)
    vp.update()

    step = 10
    vp_mode = "absolute"
    fixed_size = 1000
    dimension = "height"
    while True:
        # animator.update()
        # if animator.playing:
        #     vp.set_state(animator.current_state)

        vp.update()
        # new_frame = scale_and_fill(vp.view,1920,1080)
        new_frame, scale = fix_dimension(vp.view,fixed_size,dimension)
        cv2.imshow('Viewport Region', new_frame)
        key = cv2.waitKey(2000)

        # Exit the loop on 'q'
        if key == ord('q'):
            break

        # Move the viewport

        elif key == ord('w'):
            # viewport_y = max(rotated_height/2, viewport_y - 10)
            step = vp.h // 3
            vp.move("up",step,vp_mode)
        elif key == ord('a'):
            # viewport_x = max(rotated_width/2, viewport_x - 10)
            step = vp.w // 3
            vp.move("left",step,vp_mode)
        elif key == ord('s'):
            # viewport_y = min(image.shape[0] - (rotated_height/2), viewport_y + 10)
            step = vp.h // 3
            vp.move("down",step,vp_mode)
        elif key == ord('d'):
            # viewport_x = min(image.shape[1] - (rotated_width/2), viewport_x + 10)
            step = vp.w // 3
            vp.move("right",step,vp_mode)

        # Rotate the viewport
        elif key == ord('e'):
            vp.da -= 1
        elif key == ord('r'):
            vp.da += 1
            # vp.a = (vp.a + 5) % 360
            # print("Angle:",vp.a, "  sine:",np.sin(np.radians(vp.a)))
            # if abs(np.sin(np.radians(vp.a))) > abs(np.cos(np.radians(vp.a))):
            #     print("error")

        # Scale the viewport
        elif key == ord('+'):
            # viewport_scale += 0.1
            zs = 1.1
            if vp.h * zs <= vp.image.shape[0] and vp.w * zs <= vp.image.shape[1]:
                vp.h = int(vp.h * zs)
                vp.w = int(vp.w * zs)
            else:
                print(f"vph:{vp.h*zs} ? {vp.image.shape[0]} || vpw:{vp.w*zs} ? {vp.image.shape[1]}  ")
            # vp.h = int(min(vp.h * 1.1,1000))
            # vp.w = int(min(vp.w * 1.1,1000))
        elif key == ord('-'):
            # viewport_scale = max(0.1, viewport_scale - 0.1)
            vp.h = int(max(vp.h * 0.9,10))
            vp.w = int(max(vp.w * 0.9,10))

        elif key == ord('z'):
            vp.reset()

        # elif key == ord('v'): # Animator playpause
        #     animator.playpause()
        # elif key == ord('b'): # Add state
        #     animator.add_state(vp.get_state())
        # elif key == ord('n'): # Reset
        #     print(animator)
        # elif key == ord('m'):
        #     animator.reset()

        # elif key == ord('8'): # VP Height +
        #     vp.h += 50
        # elif key == ord('2'): # VP Height -
        #     vp.h -= 50
        # elif key == ord('6'): # VP Width +
        #     vp.w += 50
        # elif key == ord('4'): # VP Width -
        #     vp.w -= 50

    # Release resources
    cv2.destroyAllWindows()

    print("Done.")


# if __name__ == "__main__":
#     im_file = "C:/Users/nullp/Projects/map_accuracy_eval/input/420-rough-raster.png"
#     # img = cv2.imread(im_file)

#     print("done.")