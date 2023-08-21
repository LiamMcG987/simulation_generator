import os
import shutil
import customtkinter
import tkinter
import textwrap
from datetime import datetime as dt
from tkinter import filedialog

import numpy as np
from PIL import Image

from inp_populater.get_simulation_components import PopulateSimulation
from gui.generate_plot_3d import GeneratePlot
from gui.mirror_coords import mirror_coords


def file_check(file, copy_loc):
    copy_location = os.path.join(os.getcwd(), copy_loc)

    file_name_no_ext = os.path.splitext(os.path.split(file)[-1])[0]
    file_name = os.path.split(file)[-1]
    file_dir_check = os.path.join(os.getcwd(), copy_location)

    for file_dir in os.listdir(file_dir_check):
        if file_name == file_dir:
            full_path = os.path.join(file_dir_check, file_dir)
            os.remove(full_path)
            break

    shutil.copy(file, copy_location)

    return file_name_no_ext


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.load_case = None
        self.rigid_body_parts = None
        self.sbm_file = None
        self.perf_file = None
        self.bottle_mass = None
        self.translation_1, self.translation_2 = None, None

        customtkinter.set_appearance_mode('dark')
        self.title("performance_populater.py")
        self.minsize(950, 900)
        self.maxsize(1100, 1000)

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gui/images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "bmt_icon.webp")), size=(26, 26))
        self.sbm_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "sbm_bottle_image.png")),
                                                size=(30, 30))

        # create navigation frame
        self._create_navigation_frame()

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid()
        self.home_frame.columnconfigure(3, weight=1)

        self.sbm_upload_button = self._create_upload_button(self.home_frame, 'Upload SBM Data',
                                                            self.upload_sbm, [0, 1])

        self.mirror_sbm_frame = customtkinter.CTkFrame(self.home_frame, fg_color="transparent")
        self.mirror_sbm_frame.grid(column=0, row=2)

        self.mirror_sbm_data_check = customtkinter.CTkCheckBox(self.mirror_sbm_frame, text='Mirror SBM data?',
                                                               command=self.reflect_sbm)
        self.mirror_sbm_data_check.grid(column=0, row=0, pady=10)

        self.mirror_sbm_axis_combo = customtkinter.CTkComboBox(self.mirror_sbm_data_check, values=['y', 'z'],
                                                               width=15, corner_radius=5, bg_color='transparent')
        self.mirror_sbm_axis_combo.grid(column=0, row=1, padx=10, pady=10)

        self.mirror_sbm_visualiser_button = customtkinter.CTkButton(self.mirror_sbm_frame, image=self.sbm_image,
                                                                    text='', width=10,
                                                                    command=self.visualise_sbm)
        self.mirror_sbm_visualiser_button.grid(column=1, row=0, pady=10, padx=5)

        self.mesh_upload_button = self._create_upload_button(self.home_frame, 'Upload Meshed Bottle',
                                                             self.upload_perf, [0, 3])

        self.mass_entry_frame = customtkinter.CTkFrame(self.home_frame, fg_color="transparent")
        self.mass_entry_frame.grid(column=0, row=4, pady=10)

        text_var = tkinter.StringVar(value="Bottle mass (g):")
        self.mass_entry_label = customtkinter.CTkLabel(self.mass_entry_frame, textvariable=text_var)
        self.mass_entry_label.grid(column=0, row=0)
        self.mass_entry_input = customtkinter.CTkEntry(self.mass_entry_frame, width=50, justify='center',
                                                       placeholder_text='50')
        self.mass_entry_input.grid(column=1, row=0, padx=5)

        # create radio buttons
        self.case_radio = tkinter.IntVar(0)
        self.radio_button_tl = self._create_radio_button(self.home_frame, 'Top Load', 1, [0, 5])
        self.radio_button_sq = self._create_radio_button(self.home_frame, 'Squeeze', 2, [0, 6])
        self.radio_button_sl = self._create_radio_button(self.home_frame, 'Side Load', 3, [0, 7])
        self.radio_button_bul = self._create_radio_button(self.home_frame, 'Bulge', 4, [0, 8])
        self.radio_button_bur = self._create_radio_button(self.home_frame, 'Burst', 5, [0, 9])

        # create translation frame
        self.trans_matrix_frame = customtkinter.CTkFrame(self.home_frame, fg_color="transparent")
        self.trans_matrix_frame.grid(column=0, row=10, rowspan=3)

        self.trans_frame_title = self._create_label_xyz(self.trans_matrix_frame, 'Rigid Body Translations',
                                                        [0, 0], columnspan=3)
        self.trans_label_1 = self._create_label_xyz(self.trans_matrix_frame, 'Rigid Body 1', [1, 1])
        self.trans_label_2 = self._create_label_xyz(self.trans_matrix_frame, 'Rigid Body 2', [2, 1])

        self.x_label = self._create_label_xyz(self.trans_matrix_frame, 'x', [0, 2])
        self.y_label = self._create_label_xyz(self.trans_matrix_frame, 'y', [0, 3])
        self.z_label = self._create_label_xyz(self.trans_matrix_frame, 'z', [0, 4])

        self.trans_entry_rb1 = self._create_entry_xyz(self.trans_matrix_frame, [1], [2, 4])
        self.trans_entry_rb2 = self._create_entry_xyz(self.trans_matrix_frame, [2], [2, 4])

        for i in range(len(self.trans_entry_rb1)):
            self.trans_entry_rb1[i].configure(state='disabled')
            self.trans_entry_rb2[i].configure(state='disabled')

        # create file upload display
        sbm_placeholder = tkinter.StringVar(value='SBM Data')
        self.sbm_file_display = customtkinter.CTkLabel(self.home_frame, textvariable=sbm_placeholder,
                                                       justify='center', fg_color='transparent')
        self.sbm_file_display.grid(column=1, row=1, padx=10, pady=20, columnspan=3)

        perf_placeholder = tkinter.StringVar(value='Performance Data')
        self.perf_file_display = customtkinter.CTkLabel(self.home_frame, textvariable=perf_placeholder,
                                                        justify='center', fg_color='transparent')
        self.perf_file_display.grid(column=1, row=3, padx=10, pady=20, columnspan=3)

        # create canvas
        self.plot_generate_button = customtkinter.CTkButton(self.home_frame, command=self._generate_3d_plot,
                                                            text='Generate 3D Plot')
        self.plot_generate_button.grid(row=4, column=3)
        self.canvas_frame = customtkinter.CTkFrame(self.home_frame)
        self.canvas_frame.grid(row=5, column=1, rowspan=7, columnspan=3, sticky='ns')

        self.run_button = customtkinter.CTkButton(master=self.home_frame, command=self._run_process,
                                                  text='Populate Simulation')
        self.run_button.grid(column=3, row=13, padx=12, pady=10)

        self.progress_bar = customtkinter.CTkProgressBar(self.home_frame)
        self.progress_bar.grid_columnconfigure(0, weight=1)
        self.progress_bar.grid(row=14, column=1, columnspan=5, pady=10)

        self.scrollbar_frame = customtkinter.CTkFrame(self.home_frame, height=150, width=20)
        self.scrollbar_frame.grid_columnconfigure(0, weight=1)
        self.scrollbar_frame.grid(column=0, row=15, pady=10, padx=10, columnspan=5, sticky='ew')

        self.scrollbar_log = customtkinter.CTkScrollableFrame(self.scrollbar_frame, height=150)
        self.scrollbar_log.grid(column=0, row=0, sticky='ew')
        self.scrollbar_log.configure(height=10)

        # create second frame
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # select default frame
        self.select_frame_by_name("home")

        self.mainloop()

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.rb_extract_button.configure(fg_color=("gray75", "gray25") if name == "rb_extract" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "rb_extract":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()

    def _create_navigation_frame(self):
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="  3Dx Performance Generator",
                                                             image=self.logo_image,
                                                             compound="left",
                                                             font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10,
                                                   text="Simulation Generator",
                                                   fg_color="transparent", text_color=("gray10", "gray90"),
                                                   hover_color=("gray70", "gray30"),
                                                   anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.rb_extract_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                         border_spacing=10, text="Rigid Body Extractor",
                                                         fg_color="transparent", text_color=("gray10", "gray90"),
                                                         hover_color=("gray70", "gray30"),
                                                         anchor="w",
                                                         command=self.rb_extract_button_event)
        self.rb_extract_button.grid(row=2, column=0, sticky="ew")

    def home_button_event(self):
        self.select_frame_by_name("home")

    def rb_extract_button_event(self):
        self.select_frame_by_name("rb_extract")

    def get_rb_labels(self):
        labels = {
            'top load': [['Top Plate\n( + z )', 'Ground Plate\n( - z )'], ['top_plate', 'ground_plate']],
            'side load': [['Side Plate\n(Moving/ - x )', 'Side Plate\n(Fixed/ + x )'], ['side_1', 'side_2']],
            'squeeze': [['Finger\n( - y )', 'Hand/Palm\n( + y )'], ['finger', 'hand']],
            'bulge': [[None, None], [None, None]],
            'burst': [[None, None], [None, None]],
        }

        rb_labels = labels.get(self.load_case)[0]
        self.rigid_body_parts = labels.get(self.load_case)[1]
        return rb_labels

    def _create_label_xyz(self, root, text, grid_loc, columnspan=None):
        columnspan = columnspan if columnspan is not None else False

        text_var = tkinter.StringVar()
        text_var.set(text)
        label = customtkinter.CTkLabel(root, textvariable=text_var, justify='center')
        if columnspan:
            label.grid(column=grid_loc[0], row=grid_loc[1], padx=2, pady=8, columnspan=columnspan)
            label.configure(font=('helvetica underline', 16))
        else:
            label.grid(column=grid_loc[0], row=grid_loc[1], padx=2, pady=5)
        return label

    def _create_entry_xyz(self, root, grid_loc_column, grid_loc_rows):
        column = grid_loc_column
        rows = grid_loc_rows

        entries = []
        for row in np.arange(rows[0], rows[1] + 1):
            entry = customtkinter.CTkEntry(root, width=75, justify='center', placeholder_text='0')
            entry.grid(column=column, row=row, padx=5, pady=5)
            entries.append(entry)
        return entries

    def _create_radio_button(self, root, text, value, grid_loc, sticky=None):
        sticky = sticky if sticky is not None else None

        radio_button = customtkinter.CTkRadioButton(master=root, text=text,
                                                    command=self.load_case_radio,
                                                    variable=self.case_radio, value=value)
        radio_button.grid(column=grid_loc[0], row=grid_loc[1], sticky=sticky, pady=10)
        return radio_button

    def load_case_radio(self):
        cases = {
            1: 'top load',
            2: 'squeeze',
            3: 'side load',
            4: 'bulge',
            5: 'burst'
        }

        case_number = self.case_radio.get()
        load_case = cases.get(case_number)
        self.load_case = load_case
        rb_labels = self.get_rb_labels()
        self.change_label(rb_labels)

    def change_label(self, labels):
        if None in labels:
            for i in range(len(self.trans_entry_rb1)):
                self.trans_entry_rb1[i].configure(state='disabled')
                self.trans_entry_rb2[i].configure(state='disabled')
            label_1, label_2 = tkinter.StringVar(value='N/A'), tkinter.StringVar(value='N/A')
        else:
            label_1 = tkinter.StringVar(value=labels[0])
            label_2 = tkinter.StringVar(value=labels[1])

            for i in range(len(self.trans_entry_rb1)):
                self.trans_entry_rb1[i].configure(state='normal')
                self.trans_entry_rb2[i].configure(state='normal')

        self.trans_label_1.configure(textvariable=label_1)
        self.trans_label_2.configure(textvariable=label_2)

    def reflect_sbm(self):
        pass

    def visualise_sbm(self):
        mirror_coords(self.sbm_file, 'z', visualise_only=True)

    def get_bottle_mass(self):
        bottle_mass = self.mass_entry_input.get()
        self.bottle_mass = bottle_mass
        return bottle_mass

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def _create_upload_button(self, root, text, command, grid_loc, sticky=None):
        sticky = sticky if sticky is not None else None

        button_file = customtkinter.CTkButton(master=root, command=command,
                                              text=text)
        button_file.grid(column=grid_loc[0], row=grid_loc[1], sticky=sticky, pady=10, padx=30)
        return button_file

    def get_translation_matrix(self):
        translation_1, translation_2 = [], []
        for i in range(len(self.trans_entry_rb1)):
            translation_1.append(self.trans_entry_rb1[i].get())
            translation_2.append(self.trans_entry_rb2[i].get())

        self.translation_1 = translation_1
        self.translation_2 = translation_2

        return translation_1, translation_2

    def _generate_3d_plot(self):
        translation_matrix_1, translation_matrix_2 = self.get_translation_matrix()
        components = [self.rigid_body_parts, self.perf_file, self.load_case, translation_matrix_1, translation_matrix_2]
        bulge_components = [self.perf_file, self.load_case]
        if self.load_case == 'bulge' and any(not x for x in bulge_components) or any(not x for x in components):
            self.print_to_scrollbar_completed_file(
                '''
            Sections missing to generate plot.
                Ensure bottle data, load case and translations are filled in.
                ''')
        else:
            self.figure = GeneratePlot(self.canvas_frame, self.perf_file, self.rigid_body_parts,
                                       [translation_matrix_1, translation_matrix_2])

    def print_to_scrollbar_completed_file(self, text):
        scrolled_text = self.scrollbar_log
        text = textwrap.dedent(text)

        now = dt.now().strftime("%d-%m-%Y %H:%M:%S")

        label = tkinter.StringVar(value=f'{now}: {text}')
        label_widget = customtkinter.CTkLabel(master=scrolled_text, textvariable=label)
        label_widget.grid(pady=2)

    def upload_sbm(self, event=None):
        sbm_file = tkinter.filedialog.askopenfilename()
        self.sbm_file = sbm_file
        self.sbm_file_name = os.path.split(self.sbm_file)[-1]

        sbm_display = tkinter.StringVar(value=self.sbm_file_name)
        self.sbm_file_display.configure(textvariable=sbm_display, font=('Helvetica', 14))

    def upload_perf(self, event=None):
        perf_file = tkinter.filedialog.askopenfilename()
        self.perf_file = perf_file
        self.perf_file_name = os.path.split(self.perf_file)[-1]

        perf_display = tkinter.StringVar(value=self.perf_file_name)
        self.perf_file_display.configure(textvariable=perf_display, font=('Helvetica', 14))

    def _run_process(self):
        bottle_mass = self.get_bottle_mass()
        translation_matrix_1, translation_matrix_2 = self.get_translation_matrix()

        sbm_file = self.sbm_file
        perf_file = self.perf_file
        load_case = self.load_case
        components = [self.sbm_file, self.perf_file, bottle_mass, self.load_case,
                      translation_matrix_1, translation_matrix_2]

        if any(not x for x in components):
            self.print_to_scrollbar_completed_file(
                '''
            Sections missing to populate simulation file.
                Ensure bottle data, load case and translations are filled in.
                ''')
        else:
            sbm_copy_location = './inp_populater/mapping_data_inputs/'
            perf_copy_location = './part_library/bottles/'
            perf_file_no_ext = file_check(perf_file, perf_copy_location)
            sbm_file_no_ext = file_check(sbm_file, sbm_copy_location)

            self.progress_bar.set(0)
            self.progress_bar.start()
            PopulateSimulation(sbm_file_no_ext, perf_file_no_ext, load_case, bottle_mass,
                               [translation_matrix_1, translation_matrix_2])

            completed_sim_name = f'{perf_file_no_ext}_{str(load_case).replace(" ", "")}.inp'
            text = f"Simulation file generated - {completed_sim_name}. Located under simulation_inps directory."

            self.print_to_scrollbar_completed_file(text)
            self.progress_bar.set(1)


if __name__ == "__main__":
    app = App()
