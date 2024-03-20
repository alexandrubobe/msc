# -*- coding: utf-8 -*-
"""Embedding Gui Classes.

This module implements a number of classes for the display of a
tSNE or other embedding. It supports a number of selection methods
and will return selection events correcpoiding to transient, permanent
and new analysis selections

Classes:
    SelectionEvent
    EmbeddingGui
    EmbeddingViewer

"""
import io
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.widgets import RectangleSelector, LassoSelector, EllipseSelector, PolygonSelector
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
import copy
import json
import os
import string
import random
import pandas as pd

from cursors import DrawingCursors, DrawingShape, DrawingMode
from CompositeImageViewer import CompositeImageViewer


def calculate_ellipse_path(x1, y1, x2, y2):
    """This is how matplotlib draws the ellipse - thanks for the code"""

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])
    center = [x1 + (x2 - x1) / 2., y1 + (y2 - y1) / 2.]
    a = (xmax - xmin) / 2.
    b = (ymax - ymin) / 2.
    rad = np.deg2rad(np.arange(31) * 12)
    x = a * np.cos(rad) + center[0]
    y = b * np.sin(rad) + center[1]
    verts = np.stack((np.transpose(x), np.transpose(y)), 1)
    print(f"Shape verts {verts.shape}")
    return Path(verts)


class SelectionEvent(Enum):
    """The possible selection types from the EmbeddingGui"""
    TRANSIENT = 1  # For example mouse over - just views nearest points
    PERMANENT = 2  # Area selected


class EmbeddingGui(FigureCanvas):
    """This is the matplotlib based GUI for a single analysis
       It assumes the analysis is simple image data (this could be abstracted)"""

    def __init__(self, width=5, height=5):
        DrawingCursors.init_cursors()
        self.fig = Figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        mplstyle.use("fast")
        super(EmbeddingGui, self).__init__(self.fig)
        self.fig.tight_layout(pad=0)
        self.fig.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.fig.canvas.setFocus()
        self.draw_state = (DrawingMode.NoDraw, DrawingShape.NoShape)
        self.selection_mask = np.full((0, 0), False)
        self.in_selection = False
        self.facecolors = None
        self.scatter = None
        self.active_selector = None
        self.embedding_annotations = []
        self.bool_annotate = False
        self.linewidth = None
        self.setWhatsThis("""
    Select keys:
    ============
    A - select all
    D - remove selection
    I - invert selection
    
    Shaped selector keys:
    =====================
    E - ellipse
    L - lasso
    P - polygon
    R - rectangle
    Space - remove selector
    
    Selector modifier keys
    ======================
    <Shift> + <selector key> - add selection
    <Ctrl> + <selector key> - subtract selection
    
    Annotation keys:
    ===============
    N - Enter annotation mode - Blue lasso
    M - Exit annotation mode - Red lasso
    Points annotated in the current session have expanded edges.  
            """)

        plt.show(block=False)

    def get_canvas_timer(self, interval):
        """ Get a timer for this canvas widget,
            can be used to drive the embedding process"""
        return self.fig.canvas.new_timer(interval)

    def init_plot(
            self,
            embedding,
            weights,
            on_selection,
            on_close,
            landmark_orig_indexes,
            disable_select,
            top_level=False,
            labels=None,
            color_norm=None):
        """ Set the initial embedding and point weights
            to create and display the plot at iteration step=0"""

        self.top_level = top_level
        self.labels = labels
        self.color_norm = color_norm
        self.disable_select = disable_select
        self.selection_mask = np.full(weights.shape, False)

        # Plot and image definition
        # self.rect = None
        self.extent = 2.8
        self.cleanup = True
        self.rectangle_selector = None
        self.lasso_selector = None

        # Callbacks
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_over)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.fig.canvas.mpl_connect('button_press_event', self.on_start_select)
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

        # Brush support values
        self.in_selection = False
        self.in_paint = False
        self.dim_xy = (None, None)  # displacement
        self.rorg_xy = (None, None)  # The bottom left corner

        print("Show plot")
        plt.show(block=False)  # Need no block for cooperation with tkinter

        self.embedding = embedding
        self.on_selection = on_selection
        self.on_close = on_close
        self.landmark_orig_indexes = landmark_orig_indexes
        x = embedding[:, 0]
        y = embedding[:, 1]
        if self.labels is None:
            self.scatter = self.ax.scatter(
                x, y, s=weights * 8, c='b', alpha=0.4, picker=10)
        else:
            self.scatter = self.ax.scatter(
                x, y,
                s=weights * 8,
                c=self.labels,
                # TODO make color map user selectable
                cmap=plt.cm.rainbow_r,  # # pylint: disable=no-member
                norm=self.color_norm,
                alpha=0.4,
                picker=10)

        self.update_scatter_plot_limits(embedding)

        # Drawing selectors
        rectangle_selector = RectangleSelector(
            self.ax, self.on_end_rectangle_select,
            drawtype='box', useblit=True,
            button=[1, 3],
            rectprops=dict(facecolor=(1, 0, 0, 0.1), edgecolor=(1, 0, 0, 0.5), fill=False),
            minspanx=5, minspany=5,
            spancoords='pixels')
        lasso_selector = LassoSelector(
            self.ax, onselect=self.on_end_lasso_select,
            lineprops=dict(color=(1, 0, 0, 0.5)))
        ellipse_selector = EllipseSelector(
            self.ax, onselect=self.on_end_ellipse_select,
            drawtype='line', lineprops=dict(color=(1, 0, 0, 0.5)))
        polygon_selector = PolygonSelector(
            self.ax, onselect=self.on_end_lasso_select,
            lineprops=dict(color=(1, 0, 0, 0.5)))

        self.selectors = {
            DrawingShape.Lasso: lasso_selector,
            DrawingShape.Ellipse: ellipse_selector,
            DrawingShape.Rectangle: rectangle_selector,
            DrawingShape.Polygon: polygon_selector
        }
        self.set_selector()
        # force rendering of facecolors
        self.fig.canvas.draw()
        self.facecolors = self.scatter.get_facecolors()

    def quit(self):
        """Close the plot and cleanup model"""
        plt.close(self.fig)

    def handle_close(self, evt):  # pylint: disable=unused-argument
        """Cleanup at the canvas close event"""
        self.fig.canvas.stop_event_loop()
        self.on_close()

    def update_scatter_plot_limits(self, embedding):
        """Resize the plot te respect the extent of the updated embedding"""
        minp = np.amin(embedding, axis=0)
        maxp = np.amax(embedding, axis=0)

        extent_x = (maxp[0] - minp[0]) if (maxp[0] - minp[0]) > 0 else 0.1
        extent_y = (maxp[1] - minp[1]) if (maxp[1] - minp[1]) > 0 else 0.1
        self.ax.set(
            xlim=(
                minp[0] -
                0.1 *
                extent_x,
                maxp[0] +
                0.1 *
                extent_x),
            ylim=(
                minp[1] -
                0.1 *
                extent_y,
                maxp[1] +
                0.1 *
                extent_y))

    def force_refresh(self):
        """Force complete canvas redraw"""
        if (not self.facecolors is None and
                not self.scatter is None):
            self.scatter.set_facecolors(self.facecolors)
        fig_size = self.fig.get_size_inches()
        self.fig.set_size_inches(fig_size)
        self.fig.canvas.draw()
        if self.active_selector:
            self.active_selector.update()

    def update_plot(self, embedding):
        """Set new coordinated for the embedding and redraw everything"""
        self.embedding = embedding
        self.scatter.set_offsets(embedding)
        self.update_scatter_plot_limits(embedding)
        self.force_refresh()

    def set_face_colors(self, colors):
        """Colors is a numpy array of #XXXXXX
            format string colors"""
        self.facecolors = list(colors)
        self.force_refresh()

    def on_over(self, event):
        """Handle a transient selection"""
        self.fig.canvas.setFocus()
        if not self.in_selection:
            # Emit a transient selection
            # Clear the annotations
            # for annotation in self.embedding_annotations:
            #     annotation.remove()
            # self.embedding_annotations = []

            cont, index = self.scatter.contains(event)
            if cont:
                selected_indexes = list(index['ind'])
                self.on_selection(selected_indexes, SelectionEvent.TRANSIENT, self.landmark_orig_indexes,
                                  self.bool_annotate)
                # self.display_images_on_hover(selected_indexes, self.landmark_orig_indexes)
            else:
                self.on_selection([], SelectionEvent.TRANSIENT, self.landmark_orig_indexes, self.bool_annotate)

    def display_images_on_hover(self, selected_indexes, landmark_orig_indexes):
        """Display images on hover over the selected points"""
        for index in selected_indexes:
            # print(landmark_orig_indexes[index])
            x, y = self.embedding[index]
            # img_path = f"C:\\Users\\alexa\\Desktop\\frame_000000000{index%2 + 1}.jpg"  # Replace with the actual path to your image
            img_path = f"C:\\Users\\alexa\\Desktop\\EPIC-Kitchen100\\Frames_val\\{landmark_orig_indexes[index]}.jpg"
            img = plt.imread(img_path)
            imagebox = OffsetImage(img, zoom=0.5, resample=True, alpha=0.8)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
            self.embedding_annotations.append(self.ax.add_artist(ab))

    def refresh_selection(self):
        """Adjust the edge colors to reflect the selection
            and trigger the callbask"""
        xys = self.scatter.get_offsets()
        # expand the single color case to give colors for all points
        # Set default edge colors same as face
        ec = self.scatter.get_facecolors()
        if ec.shape[0] == 1:
            ec = np.tile(ec, (xys.shape[0], 1))
        if self.linewidth is None:
            lw = np.full((ec.shape[0]), 1)
        else:
            lw = copy.deepcopy(self.linewidth)
        # Set selected edge colors to dark grey and 2 thickness
        ec[self.selection_mask] = (0.05, 0.05, 0.05, 1)
        lw[self.selection_mask] = 2
        self.scatter.set_edgecolors(ec)
        self.scatter.set_linewidths(lw)
        self.fig.canvas.draw_idle()
        # Emit a permanent selection
        # return the current selection indexes
        self.on_selection(
            list(np.nonzero(
                self.selection_mask)[0]),
            SelectionEvent.PERMANENT, self.landmark_orig_indexes, self.bool_annotate)

    def on_keypress(self, event):
        """Handle the keyboard shortcuts"""
        if event.key in ['q', 'Q', 'escape']:
            self.quit()
        update = False
        if event.key in ['i', 'I']:
            self.selection_mask = ~self.selection_mask
            update = True
        if event.key in ['a', 'A']:
            self.selection_mask.fill(True)
            update = True
        if event.key in ['d', 'D']:
            self.selection_mask.fill(False)
            update = True
        if event.key in ['n', 'N']:
            annotate_indexes_landmarks = np.where(self.selection_mask)[0]
            self.bool_annotate = True
            # change lasso color to blue
            for selector in list(self.selectors.items())[:1]:
                selector[1].line.set_color((0, 0, 1, 0.5))
        if event.key in ['m', 'M']:
            self.bool_annotate = False
            # change lasso color back to red
            for selector in list(self.selectors.items())[:1]:
                selector[1].line.set_color((1, 0, 0, 0.5))
        if update:
            self.refresh_selection()
            return

        self.draw_state = DrawingCursors.get_drawstate(event.key)
        self.setCursor(DrawingCursors.cursors[self.draw_state])
        self.set_selector(self.draw_state[1])

    def in_zoom_or_pan(self):
        """Return true if the matplotlib zoom/pan function is active"""
        return bool(self.fig.canvas.toolbar.mode)

    def set_selector(self, enable=None):
        """Set the currently visible path selection widget"""
        self.active_selector = None
        for key, selector in self.selectors.items():
            # Need to also set the visibility for the polygon selector
            if key == enable:
                selector.set_active(True)
                selector.set_visible(True)
                self.active_selector = selector
            else:
                selector.set_active(False)
                selector.set_visible(False)

    def on_start_select(self, event):
        """Check that the user has started a selection and record that"""
        print(f"Event key {event.key}")

        if self.disable_select:
            return
        if not event.inaxes == self.scatter.axes:
            return

        if self.draw_state == (DrawingMode.NoDraw, DrawingShape.NoShape):
            return
        self.in_selection = True

    def on_end_rectangle_select(self, pressed, released):
        """Use the rectangle path to make a
            selection based on the selecton state"""
        x1, y1 = pressed.xdata, pressed.ydata
        x2, y2 = released.xdata, released.ydata
        path = Path(
            np.array(([x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1])))
        self.select_path(path)

    def on_end_ellipse_select(self, pressed, released):
        """Use the ellipse path to make a
            selection based on the selecton state"""
        path = calculate_ellipse_path(
            pressed.xdata,
            pressed.ydata,
            released.xdata,
            released.ydata)
        self.select_path(path)

    def on_end_lasso_select(self, verts):
        """Use the lasso or polygon path to make a
            selection based on the selecton state"""
        path = Path(verts)
        self.select_path(path)

    def select_path(self, path):
        """make a selection from a path"""
        self.in_selection = False
        xys = self.scatter.get_offsets()
        # length N bool array
        selection_mask = path.contains_points(xys)
        # print(f"selection shape {selection_mask.shape}")
        # print(f"current selection shape {self.selection_mask.shape}")
        # print(f"shape xys {xys.shape}")
        if self.draw_state[0] == DrawingMode.New:
            self.selection_mask = selection_mask
        if self.draw_state[0] == DrawingMode.Add:
            self.selection_mask = self.selection_mask | selection_mask
        if self.draw_state[0] == DrawingMode.Sub:
            self.selection_mask = self.selection_mask & ~selection_mask
        self.refresh_selection()

    def set_selection(self, indexes):
        """Set a non-interactive selection.
            This could be a saved selection or one driven
            by meta data"""
        self.selection_mask.fill(False)
        self.selection_mask[indexes] = True
        self.refresh_selection()

    def get_figure_as_buffer(self):
        """Return the scatter plot as an image in a buffer"""
        self.force_refresh()
        buf = io.BytesIO()
        extent = self.scatter.get_tightbbox(
            self.fig.canvas.get_renderer()).transformed(
            self.fig.dpi_scale_trans.inverted())

        self.fig.savefig(buf, bbox_inches=extent)
        self.force_refresh()
        return buf


class EmbeddingViewer(QtWidgets.QWidget):
    """Display an embedding gui for interaction and selection.
        In addition show the number of landmarks selected and give the
        user the chance to start a new analysis"""

    def __init__(self, parent=None):
        super(QtWidgets.QWidget, self).__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = EmbeddingGui()
        self.main_layout.addWidget(self.plot_widget)
        self.control_layout = QtWidgets.QHBoxLayout(self)
        self.landmarks_label = QtWidgets.QLabel("Landmarks selected: 0")
        self.control_layout.addWidget(self.landmarks_label)
        self.new_analysis_button = QtWidgets.QPushButton("New Analysis")
        self.new_analysis_button.clicked.connect(self.on_new_analysis)
        self.control_layout.addWidget(self.new_analysis_button)
        self.main_layout.addLayout(self.control_layout)
        self.main_layout.setStretch(0, 1)
        self.image_viewer = CompositeImageViewer()
        self.main_layout.addWidget(self.image_viewer, 0)
        self.setLayout(self.main_layout)
        self.selection_list = []

    def init_plot(
            self,
            embedding,
            weights,
            on_selection,
            on_close,
            landmark_orig_indexes,
            disable_select=False,
            top_level=False,
            labels=None,
            color_norm=None):
        self.plot_widget.init_plot(
            embedding,
            weights,
            self.on_selection,
            on_close,
            landmark_orig_indexes,
            disable_select,
            top_level,
            labels,
            color_norm)
        self.on_selection_callback = on_selection
        self.image_viewer.init_plot(None, (256, 456))

    def set_face_colors(self, color_array):
        self.plot_widget.set_face_colors(color_array)

    def set_selection(self, index_array):
        self.plot_widget.set_selection(index_array)

    @QtCore.pyqtSlot()
    def on_selection(self, indexes, selection_event, landmark_orig_indexes, bool_annotate):
        # print(f"indexes: {indexes.shape} selection size: {len(list(indexes))} perm: {permanent}")
        if selection_event is SelectionEvent.PERMANENT:
            self.selection_list = indexes
            if len(indexes) > 0 and bool_annotate:
                name_anno, done1 = QtWidgets.QInputDialog.getText(
                    self, 'Annotate the segment', 'Enter the video class:')
                if done1:
                    print("Shape of selected indexes:", np.shape(indexes))
                    if self.plot_widget.linewidth is None:
                        lw = np.full((self.plot_widget.selection_mask.shape[0]), 1)
                    else:
                        lw = self.plot_widget.linewidth
                    lw[indexes] = 5
                    self.plot_widget.linewidth = copy.deepcopy(lw)
                    self.plot_widget.scatter.set_linewidths(lw)
                    self.plot_widget.fig.canvas.draw_idle()
                    self.update_annotation_file(indexes, landmark_orig_indexes, name_anno)

            self.landmarks_label.setText(
                f"Landmarks selected: {len(self.selection_list)}")
            self.on_selection_callback(self.selection_list, False)
        else:
            # mouse over - dont change the selection list
            if len(indexes) == 0:
                # restore the previous selection
                if len(self.selection_list) > 0:
                    self.on_selection_callback(self.selection_list, False)
            else:
                image_path = f"C:\\Users\\alexa\\Desktop\\EPIC-Kitchen100\\Frames_val\\{landmark_orig_indexes[indexes[-1]]}.jpg"
                img = plt.imread(image_path)
                self.image_viewer.display_image(img)
                self.on_selection_callback(indexes, False)

    @QtCore.pyqtSlot()
    def on_new_analysis(self):
        self.on_selection_callback(self.selection_list, True)

    def set_dynamic_indexes(self, indexes):
        pass

    def get_canvas_timer(self, time):
        """Return a timer that can be used with the plot canvas"""
        return self.plot_widget.get_canvas_timer(time)

    def update_plot(self, embedding):
        """ Used in the embedding iteration phase when the xys are
        being calculated"""
        self.plot_widget.update_plot(embedding)

    def force_refresh(self):
        self.plot_widget.force_refresh()

    def get_figure_as_buffer(self):
        return self.plot_widget.get_figure_as_buffer()

    def set_static_mask(self, mask):
        """ Accept 1D num py array of 0 or 1.0 mask values """
        # self.image_widget.set_selection_alpha(mask)

    def update_annotation_file(self, indexes, landmark_orig_indexes, name_anno):
        path_to_placeholder_file = "C:\\Users\\alexa\\Desktop\\EPIC-Kitchen100\\Val_set_hsne\\annotation_file_placeholder.json"
        path_to_annotation_file = "C:\\Users\\alexa\\Desktop\\EPIC-Kitchen100\\Val_set_hsne\\annotation_file.json"
        path_to_metadata = "C:\\Users\\alexa\\Desktop\\EPIC-Kitchen100\\Val_set_hsne\\anno_val_set_segm.csv"
        interval_features = 0.5333
        dataframe = pd.read_csv(path_to_metadata, header=0)
        number_of_videos = len(np.unique(dataframe["video_label"]))
        video_names = np.unique(dataframe["video_label"])
        placeholder_data = None
        with open(path_to_placeholder_file, "r") as file:
            placeholder_data = json.load(file)
        current_data = None
        if os.path.isfile(path_to_annotation_file):
            with open(path_to_annotation_file, "r") as file:
                current_data = json.load(file)
        actual_data = None
        with open(path_to_annotation_file, "w") as file:
            if current_data is None:
                actual_data = placeholder_data
                actual_data["project"]["vid_list"] = [f"{i}" for i in range(1, number_of_videos + 1)]
                for i in range(1, number_of_videos + 1):
                    actual_data["file"][f"{i}"] = {}
                    actual_data["file"][f"{i}"]["fid"] = f"{i}"
                    actual_data["file"][f"{i}"]["fname"] = video_names[i - 1]
                    actual_data["file"][f"{i}"]["type"] = 4
                    actual_data["file"][f"{i}"]["loc"] = 1
                    actual_data["file"][f"{i}"]["src"] = ""
                for index in indexes:
                    id_string = random_generator_string()
                    video = dataframe["video_label"][landmark_orig_indexes[index]]
                    i = np.where(video_names == video)[0][0] + 1
                    moment = (landmark_orig_indexes[index] - np.where(dataframe["video_label"] == video)[0][0]) * interval_features
                    actual_data["metadata"][f"{i}_{id_string}"] = {}
                    actual_data["metadata"][f"{i}_{id_string}"]["vid"] = f"{i}"
                    actual_data["metadata"][f"{i}_{id_string}"]["flg"] = 0
                    actual_data["metadata"][f"{i}_{id_string}"]["z"] = [moment, moment + interval_features]
                    actual_data["metadata"][f"{i}_{id_string}"]["xy"] = []
                    actual_data["metadata"][f"{i}_{id_string}"]["av"] = {}
                    actual_data["metadata"][f"{i}_{id_string}"]["av"]["1"] = name_anno
                for i in range(1, number_of_videos + 1):
                    actual_data["view"][f"{i}"] = {}
                    actual_data["view"][f"{i}"]["fid_list"] = [f"{i}"]
            else:
                actual_data = current_data
                for index in indexes:
                    id_string = random_generator_string()
                    video = dataframe["video_label"][landmark_orig_indexes[index]]
                    i = np.where(video_names == video)[0][0] + 1
                    moment = (landmark_orig_indexes[index] - np.where(dataframe["video_label"] == video)[0][0]) * interval_features
                    actual_data["metadata"][f"{i}_{id_string}"] = {}
                    actual_data["metadata"][f"{i}_{id_string}"]["vid"] = f"{i}"
                    actual_data["metadata"][f"{i}_{id_string}"]["flg"] = 0
                    actual_data["metadata"][f"{i}_{id_string}"]["z"] = [moment, moment + interval_features]
                    actual_data["metadata"][f"{i}_{id_string}"]["xy"] = []
                    actual_data["metadata"][f"{i}_{id_string}"]["av"] = {}
                    actual_data["metadata"][f"{i}_{id_string}"]["av"]["1"] = name_anno
            json.dump(actual_data, file)


def random_generator_string():
    s = string.digits + string.ascii_letters
    rand_string = ''.join(random.sample(s, 8))
    return rand_string



