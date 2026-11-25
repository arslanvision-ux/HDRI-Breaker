"""
Custom themed File Manager dialog for HDRI Maker / HDRI Breaker.

Provides clearly labeled top toolbar navigation buttons (fixing invisible/tiny
standard Qt icons on Linux/Houdini dark themes) and persistent folder bookmarks
stored in ~/.hdri_maker_bookmarks.json.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from .qt_compat import QtCore, QtGui, QtWidgets, Qt, exec_dialog

try:
    import hou  # type: ignore
except ImportError:
    hou = None


BOOKMARKS_PATH = os.path.join(os.path.expanduser("~"), ".hdri_maker_bookmarks.json")


def _get_filesystem_model_cls():
    """Return QFileSystemModel across PySide2 (QtWidgets) and PySide6 (QtGui/QtWidgets)."""
    return getattr(QtWidgets, "QFileSystemModel", None) or getattr(QtGui, "QFileSystemModel", None)


def load_bookmarks_data(path: str = BOOKMARKS_PATH) -> Dict[str, object]:
    """Load saved folder bookmarks and last_dir from disk."""
    default: Dict[str, object] = {"bookmarks": [], "last_dir": ""}
    if not path or not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return default
        seen = set()
        clean_bookmarks: List[Dict[str, str]] = []
        for item in raw.get("bookmarks", []):
            if isinstance(item, str):
                b_path = os.path.normpath(item)
                b_name = os.path.basename(b_path) or b_path
            elif isinstance(item, dict):
                b_path = os.path.normpath(str(item.get("path", "")).strip())
                b_name = str(item.get("name", "")).strip() or os.path.basename(b_path) or b_path
            else:
                continue
            if not b_path or b_path in seen:
                continue
            seen.add(b_path)
            clean_bookmarks.append({"name": b_name, "path": b_path})
        last_dir = str(raw.get("last_dir", "") or "").strip()
        return {"bookmarks": clean_bookmarks, "last_dir": last_dir}
    except Exception:
        return default


def save_bookmarks_data(
    bookmarks: List[Dict[str, str]],
    last_dir: str = "",
    path: str = BOOKMARKS_PATH,
) -> bool:
    """Persist folder bookmarks and last_dir to disk."""
    if not path:
        return False
    try:
        parent_dir = os.path.dirname(os.path.abspath(path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        payload = {
            "bookmarks": [
                {
                    "name": str(b.get("name", "") or os.path.basename(b.get("path", "")) or b.get("path", "")),
                    "path": os.path.normpath(str(b.get("path", ""))),
                }
                for b in bookmarks
                if isinstance(b, dict) and str(b.get("path", "")).strip()
            ],
            "last_dir": str(last_dir or "").strip(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return True
    except Exception:
        return False


def parse_filter_string(filter_str: str) -> List[Tuple[str, List[str]]]:
    """Parse a Qt file filter string into [(label, [glob_patterns...]), ...]."""
    if not filter_str or not filter_str.strip():
        return [("All Files (*)", [])]
    parsed: List[Tuple[str, List[str]]] = []
    for part in filter_str.split(";;"):
        label = part.strip()
        if not label:
            continue
        match = re.search(r"\(([^)]+)\)", label)
        patterns: List[str] = []
        if match:
            tokens = match.group(1).strip().split()
            for tok in tokens:
                tok = tok.strip()
                if tok and tok not in ("*", "*.*"):
                    patterns.append(tok)
            if any(t in ("*", "*.*") for t in tokens):
                patterns = []
        parsed.append((label, patterns))
    return parsed or [("All Files (*)", [])]


class HDRIFileDialog(QtWidgets.QDialog):
    """Enhanced file dialog with labeled top toolbar icons and persistent folder bookmarks."""

    MODE_OPEN_ONE = "open_one"
    MODE_OPEN_MANY = "open_many"
    MODE_SAVE = "save"

    def __init__(
        self,
        parent=None,
        caption: str = "Select File",
        initial_path: str = "",
        filter_str: str = "All Files (*)",
        mode: str = MODE_OPEN_ONE,
        bookmarks_path: str = BOOKMARKS_PATH,
    ):
        super().__init__(parent)
        self.setWindowTitle(caption)
        self.resize(940, 600)
        self.setModal(True)

        self._mode = mode
        self._bookmarks_path = bookmarks_path
        self._selected_files: List[str] = []
        self._selected_filter: str = ""
        self._history: List[str] = []
        self._history_idx: int = -1
        self._navigating_history = False

        data = load_bookmarks_data(self._bookmarks_path)
        self._bookmarks: List[Dict[str, str]] = list(data.get("bookmarks", []))
        saved_last_dir = str(data.get("last_dir", "") or "").strip()

        start_dir, default_filename = self._resolve_initial_target(initial_path, saved_last_dir)
        self._filters = parse_filter_string(filter_str)

        self._build_ui(default_filename)
        self._apply_dialog_style(parent)
        self._populate_quick_places()
        self._refresh_bookmarks_list()
        self._apply_name_filter(0)
        self.navigate_to(start_dir, record_history=True)

    @staticmethod
    def _resolve_initial_target(initial_path: str, saved_last_dir: str) -> Tuple[str, str]:
        """Resolve starting directory and pre-filled filename."""
        raw = (initial_path or "").strip()
        filename = ""
        target_dir = ""
        if raw:
            expanded = os.path.expanduser(raw)
            if os.path.isdir(expanded):
                target_dir = expanded
            else:
                parent = os.path.dirname(expanded)
                base = os.path.basename(expanded)
                if parent and os.path.isdir(parent):
                    target_dir = parent
                    filename = base
                elif base and not parent:
                    filename = base
        if not target_dir and saved_last_dir and os.path.isdir(saved_last_dir):
            target_dir = saved_last_dir
        if not target_dir:
            if hou is not None:
                try:
                    hip = hou.getenv("HIP", "")
                    if hip and os.path.isdir(hip):
                        target_dir = hip
                except Exception:
                    pass
        if not target_dir or not os.path.isdir(target_dir):
            target_dir = os.path.expanduser("~")
        if not os.path.isdir(target_dir):
            target_dir = os.getcwd()
        return os.path.normpath(target_dir), filename

    def _build_ui(self, default_filename: str) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- Top Toolbar with explicit clear labels & tooltips ---
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_back = QtWidgets.QPushButton("◀ Back")
        self.btn_back.setToolTip("Go to previous visited folder (Alt+Left)")
        self.btn_back.clicked.connect(self.go_back)

        self.btn_forward = QtWidgets.QPushButton("▶ Fwd")
        self.btn_forward.setToolTip("Go to next visited folder (Alt+Right)")
        self.btn_forward.clicked.connect(self.go_forward)

        self.btn_up = QtWidgets.QPushButton("▲ Up")
        self.btn_up.setToolTip("Go to parent folder (Alt+Up)")
        self.btn_up.clicked.connect(self.go_up)

        self.btn_home = QtWidgets.QPushButton("⌂ Home")
        self.btn_home.setToolTip("Go to user Home directory")
        self.btn_home.clicked.connect(self.go_home)

        self.btn_new_folder = QtWidgets.QPushButton("+ New Folder")
        self.btn_new_folder.setToolTip("Create a new folder in the current directory")
        self.btn_new_folder.clicked.connect(self.create_new_folder)

        self.btn_bookmark = QtWidgets.QPushButton("★ Bookmark")
        self.btn_bookmark.setToolTip("Save current folder to Bookmarks sidebar")
        self.btn_bookmark.clicked.connect(self.add_current_bookmark)

        self.btn_refresh = QtWidgets.QPushButton("↻ Refresh")
        self.btn_refresh.setToolTip("Refresh directory listing")
        self.btn_refresh.clicked.connect(self.refresh_view)

        self.btn_hidden = QtWidgets.QPushButton("Show Hidden")
        self.btn_hidden.setCheckable(True)
        self.btn_hidden.setToolTip("Toggle hidden files and dot-folders")
        self.btn_hidden.toggled.connect(self._on_toggle_hidden)

        for btn in (
            self.btn_back,
            self.btn_forward,
            self.btn_up,
            self.btn_home,
            self.btn_new_folder,
            self.btn_bookmark,
            self.btn_refresh,
            self.btn_hidden,
        ):
            btn.setMinimumHeight(28)
            toolbar.addWidget(btn)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        # --- Address / Look in Bar ---
        addr_row = QtWidgets.QHBoxLayout()
        addr_row.setSpacing(6)
        addr_row.addWidget(QtWidgets.QLabel("Look in:"))
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Enter folder path and press Enter...")
        self.path_edit.returnPressed.connect(self._on_path_entered)
        addr_row.addWidget(self.path_edit, 1)

        self.btn_go = QtWidgets.QPushButton("Go")
        self.btn_go.setToolTip("Navigate to the path entered in the address bar")
        self.btn_go.setMinimumHeight(26)
        self.btn_go.clicked.connect(self._on_path_entered)
        addr_row.addWidget(self.btn_go)
        root.addLayout(addr_row)

        # --- Main Splitter: Left Bookmarks Sidebar + Right File Tree ---
        splitter = QtWidgets.QSplitter(Qt.Horizontal)

        sidebar_widget = QtWidgets.QWidget()
        sidebar_widget.setMinimumWidth(200)
        sidebar_widget.setMaximumWidth(320)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(6)

        places_lbl = QtWidgets.QLabel("Quick Places")
        places_lbl.setStyleSheet("font-weight: bold;")
        sidebar_layout.addWidget(places_lbl)

        self.places_list = QtWidgets.QListWidget()
        self.places_list.setMaximumHeight(150)
        self.places_list.itemClicked.connect(self._on_place_clicked)
        sidebar_layout.addWidget(self.places_list)

        bm_header = QtWidgets.QHBoxLayout()
        bm_lbl = QtWidgets.QLabel("Saved Bookmarks")
        bm_lbl.setStyleSheet("font-weight: bold;")
        bm_header.addWidget(bm_lbl)
        bm_header.addStretch(1)

        self.btn_add_bm = QtWidgets.QPushButton("+ Add")
        self.btn_add_bm.setToolTip("Bookmark current folder")
        self.btn_add_bm.setFixedHeight(22)
        self.btn_add_bm.clicked.connect(self.add_current_bookmark)
        bm_header.addWidget(self.btn_add_bm)

        self.btn_rem_bm = QtWidgets.QPushButton("- Remove")
        self.btn_rem_bm.setToolTip("Remove selected bookmark")
        self.btn_rem_bm.setFixedHeight(22)
        self.btn_rem_bm.clicked.connect(self.remove_selected_bookmark)
        bm_header.addWidget(self.btn_rem_bm)
        sidebar_layout.addLayout(bm_header)

        self.bookmarks_list = QtWidgets.QListWidget()
        self.bookmarks_list.itemClicked.connect(self._on_bookmark_clicked)
        sidebar_layout.addWidget(self.bookmarks_list, 1)

        splitter.addWidget(sidebar_widget)

        # File System Model + Tree View
        fs_model_cls = _get_filesystem_model_cls()
        self.fs_model = fs_model_cls(self)
        self._apply_dir_filters(show_hidden=False)
        self.fs_model.setNameFilterDisables(False)
        self.fs_model.setRootPath("")

        self.tree_view = QtWidgets.QTreeView()
        self.tree_view.setModel(self.fs_model)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.sortByColumn(0, Qt.AscendingOrder)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setRootIsDecorated(False)
        self.tree_view.setItemsExpandable(False)
        if self._mode == self.MODE_OPEN_MANY:
            self.tree_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
            self.tree_view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        header = self.tree_view.header()
        try:
            resize_enum = getattr(QtWidgets.QHeaderView, "ResizeMode", QtWidgets.QHeaderView)
            header.setSectionResizeMode(0, resize_enum.Stretch)
            header.setSectionResizeMode(1, resize_enum.ResizeToContents)
            header.setSectionResizeMode(2, resize_enum.ResizeToContents)
            header.setSectionResizeMode(3, resize_enum.ResizeToContents)
        except Exception:
            pass

        self.tree_view.doubleClicked.connect(self._on_tree_double_clicked)
        self.tree_view.selectionModel().selectionChanged.connect(self._on_tree_selection_changed)
        splitter.addWidget(self.tree_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 690])
        root.addWidget(splitter, 1)

        # --- Bottom Bar: File name, Filter, Accept/Cancel ---
        bottom_grid = QtWidgets.QGridLayout()
        bottom_grid.setSpacing(6)

        bottom_grid.addWidget(QtWidgets.QLabel("File name:"), 0, 0)
        self.filename_edit = QtWidgets.QLineEdit(default_filename)
        self.filename_edit.returnPressed.connect(self._on_accept_clicked)
        bottom_grid.addWidget(self.filename_edit, 0, 1)

        action_text = "Save" if self._mode == self.MODE_SAVE else "Open"
        self.btn_accept = QtWidgets.QPushButton(action_text)
        self.btn_accept.setObjectName("hdriPrimaryBtn")
        self.btn_accept.setMinimumWidth(96)
        self.btn_accept.setMinimumHeight(28)
        self.btn_accept.clicked.connect(self._on_accept_clicked)
        bottom_grid.addWidget(self.btn_accept, 0, 2)

        bottom_grid.addWidget(QtWidgets.QLabel("Files of type:"), 1, 0)
        self.filter_combo = QtWidgets.QComboBox()
        for label, _patterns in self._filters:
            self.filter_combo.addItem(label)
        self.filter_combo.currentIndexChanged.connect(self._apply_name_filter)
        bottom_grid.addWidget(self.filter_combo, 1, 1)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setMinimumWidth(96)
        self.btn_cancel.setMinimumHeight(28)
        self.btn_cancel.clicked.connect(self.reject)
        bottom_grid.addWidget(self.btn_cancel, 1, 2)

        root.addLayout(bottom_grid)

        # Keyboard shortcuts for Back / Forward / Up (PySide2: QtWidgets, PySide6: QtGui)
        qshortcut_cls = getattr(QtGui, "QShortcut", None) or getattr(QtWidgets, "QShortcut", None)
        if qshortcut_cls is not None:
            qshortcut_cls(QtGui.QKeySequence("Alt+Left"), self, activated=self.go_back)
            qshortcut_cls(QtGui.QKeySequence("Alt+Right"), self, activated=self.go_forward)
            qshortcut_cls(QtGui.QKeySequence("Alt+Up"), self, activated=self.go_up)

    def _apply_dialog_style(self, parent) -> None:
        """Ensure crisp contrast and visible button labels regardless of host theme."""
        base_css = """
        QDialog {
            background-color: #1b1d22;
            color: #f0f4f8;
        }
        QLabel {
            color: #f0f4f8;
        }
        QLineEdit, QComboBox, QListWidget, QTreeView {
            background-color: #15171c;
            color: #f0f4f8;
            border: 1px solid #444c5c;
            border-radius: 3px;
            padding: 3px 6px;
        }
        QTreeView::item:selected, QListWidget::item:selected {
            background-color: #f59e0b;
            color: #111318;
        }
        QPushButton {
            background-color: #2e3440;
            color: #f0f4f8;
            border: 1px solid #525c70;
            border-radius: 3px;
            padding: 3px 10px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #3b4252;
            border-color: #f59e0b;
        }
        QPushButton:checked {
            background-color: #f59e0b;
            color: #111318;
            border-color: #f59e0b;
        }
        QPushButton:disabled {
            color: #6b7280;
            background-color: #22252c;
            border-color: #343944;
        }
        QPushButton#hdriPrimaryBtn {
            background-color: #f59e0b;
            color: #111318;
            border: 1px solid #f59e0b;
            font-weight: bold;
        }
        QPushButton#hdriPrimaryBtn:hover {
            background-color: #fbbf24;
        }
        QHeaderView::section {
            background-color: #22252c;
            color: #d1d5db;
            border: 1px solid #343944;
            padding: 4px 6px;
        }
        """
        self.setStyleSheet(base_css)

    def _apply_dir_filters(self, show_hidden: bool) -> None:
        qdir = QtCore.QDir
        flags = qdir.AllDirs | qdir.Files | qdir.NoDotAndDotDot
        if show_hidden:
            flags |= qdir.Hidden
        self.fs_model.setFilter(flags)

    def _on_toggle_hidden(self, checked: bool) -> None:
        self._apply_dir_filters(show_hidden=bool(checked))

    def _apply_name_filter(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._filters):
            self.fs_model.setNameFilters([])
            return
        label, patterns = self._filters[idx]
        self._selected_filter = label
        self.fs_model.setNameFilters(patterns)

    def _populate_quick_places(self) -> None:
        self.places_list.clear()
        places: List[Tuple[str, str]] = []
        home = os.path.expanduser("~")
        if os.path.isdir(home):
            places.append(("⌂ Home", home))
        desktop = os.path.join(home, "Desktop")
        if os.path.isdir(desktop):
            places.append(("Desktop", desktop))
        if hou is not None:
            try:
                hip = hou.getenv("HIP", "")
                if hip and os.path.isdir(hip):
                    places.append(("$HIP (Scene)", hip))
            except Exception:
                pass
        if os.name == "nt":
            for letter in "CDEFGH":
                drv = f"{letter}:\\"
                if os.path.isdir(drv):
                    places.append((f"Drive ({letter}:)", drv))
        else:
            places.append(("Root (/)", "/"))

        for label, p in places:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(Qt.UserRole, os.path.normpath(p))
            item.setToolTip(p)
            self.places_list.addItem(item)

    def _refresh_bookmarks_list(self) -> None:
        self.bookmarks_list.clear()
        for bm in self._bookmarks:
            b_path = bm.get("path", "")
            b_name = bm.get("name", "") or os.path.basename(b_path) or b_path
            item = QtWidgets.QListWidgetItem(f"★ {b_name}")
            item.setData(Qt.UserRole, b_path)
            item.setToolTip(b_path)
            self.bookmarks_list.addItem(item)

    def current_dir(self) -> str:
        return os.path.normpath(self.path_edit.text().strip() or os.path.expanduser("~"))

    def navigate_to(self, dir_path: str, record_history: bool = True) -> bool:
        if not dir_path:
            return False
        norm = os.path.normpath(os.path.expanduser(dir_path))
        if not os.path.isdir(norm):
            return False
        idx = self.fs_model.index(norm)
        if not idx.isValid():
            return False
        self.tree_view.setRootIndex(idx)
        self.path_edit.setText(norm)

        if record_history and not self._navigating_history:
            if self._history_idx < len(self._history) - 1:
                self._history = self._history[: self._history_idx + 1]
            if not self._history or self._history[-1] != norm:
                self._history.append(norm)
                self._history_idx = len(self._history) - 1

        self._update_nav_buttons()
        save_bookmarks_data(self._bookmarks, last_dir=norm, path=self._bookmarks_path)
        return True

    def _update_nav_buttons(self) -> None:
        self.btn_back.setEnabled(self._history_idx > 0)
        self.btn_forward.setEnabled(0 <= self._history_idx < len(self._history) - 1)
        cur = self.current_dir()
        parent = os.path.dirname(cur)
        self.btn_up.setEnabled(bool(parent and parent != cur))

    def go_back(self) -> None:
        if self._history_idx > 0:
            self._history_idx -= 1
            self._navigating_history = True
            try:
                self.navigate_to(self._history[self._history_idx], record_history=False)
            finally:
                self._navigating_history = False

    def go_forward(self) -> None:
        if 0 <= self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            self._navigating_history = True
            try:
                self.navigate_to(self._history[self._history_idx], record_history=False)
            finally:
                self._navigating_history = False

    def go_up(self) -> None:
        cur = self.current_dir()
        parent = os.path.dirname(cur)
        if parent and parent != cur:
            self.navigate_to(parent)

    def go_home(self) -> None:
        self.navigate_to(os.path.expanduser("~"))

    def refresh_view(self) -> None:
        cur = self.current_dir()
        self.navigate_to(cur, record_history=False)

    def create_new_folder(self) -> None:
        cur = self.current_dir()
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "New Folder",
            f"Create folder inside:\n{cur}",
        )
        if not ok or not name.strip():
            return
        new_path = os.path.join(cur, name.strip())
        try:
            os.makedirs(new_path, exist_ok=True)
            self.navigate_to(new_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "New Folder Error", f"Could not create folder:\n{exc}")

    def add_current_bookmark(self) -> None:
        cur = self.current_dir()
        if not os.path.isdir(cur):
            return
        for bm in self._bookmarks:
            if os.path.normpath(bm.get("path", "")) == cur:
                return
        label = os.path.basename(cur) or cur
        self._bookmarks.append({"name": label, "path": cur})
        save_bookmarks_data(self._bookmarks, last_dir=cur, path=self._bookmarks_path)
        self._refresh_bookmarks_list()

    def remove_selected_bookmark(self) -> None:
        row = self.bookmarks_list.currentRow()
        if row < 0 or row >= len(self._bookmarks):
            return
        self._bookmarks.pop(row)
        save_bookmarks_data(self._bookmarks, last_dir=self.current_dir(), path=self._bookmarks_path)
        self._refresh_bookmarks_list()

    def _on_path_entered(self) -> None:
        self.navigate_to(self.path_edit.text().strip())

    def _on_place_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        target = item.data(Qt.UserRole)
        if target:
            self.navigate_to(str(target))

    def _on_bookmark_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        target = item.data(Qt.UserRole)
        if target:
            self.navigate_to(str(target))

    def _on_tree_selection_changed(self, _selected, _deselected) -> None:
        indexes = self.tree_view.selectionModel().selectedRows(0)
        if not indexes:
            return
        if self._mode == self.MODE_OPEN_MANY:
            names = []
            for idx in indexes:
                if not self.fs_model.isDir(idx):
                    names.append(self.fs_model.fileName(idx))
            if names:
                self.filename_edit.setText("; ".join(names))
        else:
            idx = indexes[0]
            if not self.fs_model.isDir(idx):
                self.filename_edit.setText(self.fs_model.fileName(idx))

    def _on_tree_double_clicked(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        if self.fs_model.isDir(index):
            self.navigate_to(self.fs_model.filePath(index))
        else:
            self._selected_files = [os.path.normpath(self.fs_model.filePath(index))]
            self.accept()

    def _on_accept_clicked(self) -> None:
        cur = self.current_dir()
        if self._mode == self.MODE_OPEN_MANY:
            indexes = self.tree_view.selectionModel().selectedRows(0)
            files = [
                os.path.normpath(self.fs_model.filePath(idx))
                for idx in indexes
                if not self.fs_model.isDir(idx)
            ]
            if files:
                self._selected_files = files
                self.accept()
                return

        text = self.filename_edit.text().strip()
        if not text:
            return

        # If user typed a directory path into File name box, navigate there
        candidate = text if os.path.isabs(text) else os.path.join(cur, text)
        candidate = os.path.normpath(os.path.expanduser(candidate))
        if os.path.isdir(candidate):
            self.navigate_to(candidate)
            self.filename_edit.clear()
            return

        if self._mode == self.MODE_SAVE:
            self._selected_files = [candidate]
            self.accept()
            return

        if os.path.isfile(candidate):
            self._selected_files = [candidate]
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "File Not Found", f"File does not exist:\n{candidate}")

    def selected_files(self) -> List[str]:
        return list(self._selected_files)

    def selected_filter(self) -> str:
        return self._selected_filter

    @classmethod
    def get_open_file_name(
        cls,
        parent=None,
        caption: str = "Open File",
        initial_path: str = "",
        filter_str: str = "All Files (*)",
    ) -> Tuple[str, str]:
        dlg = cls(
            parent=parent,
            caption=caption,
            initial_path=initial_path,
            filter_str=filter_str,
            mode=cls.MODE_OPEN_ONE,
        )
        if exec_dialog(dlg) == QtWidgets.QDialog.Accepted and dlg.selected_files():
            return dlg.selected_files()[0], dlg.selected_filter()
        return "", ""

    @classmethod
    def get_open_file_names(
        cls,
        parent=None,
        caption: str = "Open Files",
        initial_path: str = "",
        filter_str: str = "All Files (*)",
    ) -> Tuple[List[str], str]:
        dlg = cls(
            parent=parent,
            caption=caption,
            initial_path=initial_path,
            filter_str=filter_str,
            mode=cls.MODE_OPEN_MANY,
        )
        if exec_dialog(dlg) == QtWidgets.QDialog.Accepted and dlg.selected_files():
            return dlg.selected_files(), dlg.selected_filter()
        return [], ""

    @classmethod
    def get_save_file_name(
        cls,
        parent=None,
        caption: str = "Save File",
        initial_path: str = "",
        filter_str: str = "All Files (*)",
    ) -> Tuple[str, str]:
        dlg = cls(
            parent=parent,
            caption=caption,
            initial_path=initial_path,
            filter_str=filter_str,
            mode=cls.MODE_SAVE,
        )
        if exec_dialog(dlg) == QtWidgets.QDialog.Accepted and dlg.selected_files():
            return dlg.selected_files()[0], dlg.selected_filter()
        return "", ""
