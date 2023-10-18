import tkinter as tk
root = tk.Tk()
def on_enter(event=None):
    text = entry.get()
    popup_window.destroy()
    print("Entered text:", text)
    root.destroy()

def show_popup(default_text=""):
    global popup_window,entry
    popup_window = tk.Toplevel(root)
    popup_window.title("Enter Reference Point Label")

    entry = tk.Entry(popup_window)
    entry.insert(0, default_text)  # Default text in the entry widget
    entry.select_range(0, 'end')     # Highlight all text
    entry.focus_set()                # Set focus to the entry widget
    entry.bind('<Return>', on_enter)

    enter_button = tk.Button(popup_window, text="Enter", command=on_enter)

    entry.pack(padx=10, pady=10)
    enter_button.pack(pady=10)


# root.title("Main Window")
root.withdraw()
# popup_button = tk.Button(root, text="Show Popup", command=show_popup)
# popup_button.pack(padx=10, pady=10)
show_popup("Test")

root.mainloop()

