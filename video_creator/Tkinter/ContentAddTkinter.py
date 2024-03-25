import tkinter as tk
import json
import uuid
import os

def validate_text(event):
    text_content_description = description_entry.get("1.0", "end-1c").strip()
    text_content_title = title_entry.get().strip()
    text_content_tags = tags_entry.get().strip()
    if len(text_content_description) > 0:
        submit_button.config(state=tk.NORMAL)
    else:
        submit_button.config(state=tk.DISABLED)

    update_word_count()

def update_word_count():
    description_text = description_entry.get("1.0", "end-1c")
    word_count = len(description_text.split())
    word_count_label.config(text=f"Word Count: {word_count} , total time: {word_count * 0.370} seconds")

root = tk.Tk()
root.title("Add Future Items")

# main container
frame = tk.Frame(root, bg='white', width=500, height=500)
frame.pack(fill='both', expand='true', padx=50, pady=50)

# title
title_label = tk.Label(frame, text="Title")
title_label.pack()
title_entry = tk.Entry(frame, width=50, bg='white', fg='black', font=('Arial', 12), validate="key")
title_entry.pack()

# description
description_label = tk.Label(frame, text="Text")
description_label.pack()
description_entry = tk.Text(frame, wrap='word', width=50, height=5, bg='white', fg='black', font=('Arial', 12))
description_entry.pack()

description_entry.bind("<Key>", validate_text)

# Word count label
word_count_label = tk.Label(frame, text="Word Count: 0")
word_count_label.pack()

tags_label = tk.Label(frame, text="Tags (comma separated)")
tags_label.pack()
tags_entry = tk.Entry(frame, width=50, bg='white', fg='black', font=('Arial', 12), validate="key")
tags_entry.pack()
example_tags_label = tk.Label(frame, text="Example -> tag1, tag2, tag3")
example_tags_label.pack()

def split_tags():
    tags_string = tags_entry.get()
    tags_array = [tag.strip() for tag in tags_string.split(",")]
    return tags_array

# form button
submit_button = tk.Button(frame, text="Submit", bg='blue', fg='white', font=('Arial', 12), state=tk.DISABLED)

tags_array = []
title = ""
description = ""

def generate_json(title, description, tags):
    print("Title: " + title)
    print("Description: " + description)
    print("Tags: " + str(tags))
    json_data = {
        "title": title,
        "description": description,
        "tags": tags
    }
    print(json.dumps(json_data, indent=1))
    return json_data



def submit_form():
    global title, description, tags_array
    title = title_entry.get()
    description = description_entry.get("1.0", 'end-1c')
    tags_array = split_tags()
    root.destroy()
    # json data
    json_data = generate_json(title, description, tags_array)
    random_code = str(uuid.uuid4())

    output_path_name = "MetaData_" + random_code + ".json"
    outfile_path = os.path.dirname(os.path.abspath(__file__))
    outfile_path = os.path.dirname(outfile_path)
    outfile_path = os.path.join(outfile_path, 'Metadata')

    print(outfile_path)
    with open(os.path.join(outfile_path, output_path_name), 'w') as outfile:
        json.dump(json_data, outfile, indent=2)


submit_button.config(command=submit_form)
submit_button.pack()

root.mainloop()

