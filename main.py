import sys
import os
import time
import numpy as np
import faiss
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QFrame, QRadioButton, QButtonGroup, QSizePolicy
)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt

# -------------------------------
# Tile image mapping functions
# -------------------------------
def get_tile_path(tile_code):
    digit = tile_code[0]
    letter = tile_code[1].lower()
    if letter == 'm':
        return os.path.join("tiles", f"Man{digit}.svg")
    elif letter == 'p':
        return os.path.join("tiles", f"Pin{digit}.svg")
    elif letter == 's':
        return os.path.join("tiles", f"Sou{digit}.svg")
    elif letter == 'd':
        dragons = {'1': 'Haku.svg', '2': 'Hatsu.svg', '3': 'Chun.svg'}
        return os.path.join("tiles", dragons.get(digit, ""))
    elif letter == 'w':
        winds = {'3': 'Ton.svg', '2': 'Nan.svg', '4': 'Shaa.svg', '1': 'Pei.svg'}
        return os.path.join("tiles", winds.get(digit, ""))
    return None

def parse_hand(hand_str):
    tiles = []
    for i in range(0, len(hand_str), 2):
        tile_code = hand_str[i:i+2]
        if len(tile_code) < 2:
            continue
        path = get_tile_path(tile_code)
        if path:
            tiles.append(path)
    return tiles

# -------------------------------
# Conversion functions between hand string and 34-dim vector
# -------------------------------
def hand_to_vector(hand_str):
    vec = np.zeros(34, dtype='float32')
    for i in range(0, len(hand_str), 2):
        tile = hand_str[i:i+2]
        if len(tile) < 2:
            continue
        digit, suit = tile[0], tile[1].lower()
        if suit == 'm':
            idx = int(digit) - 1
        elif suit == 'p':
            idx = 9 + int(digit) - 1
        elif suit == 's':
            idx = 18 + int(digit) - 1
        elif suit == 'w':
            mapping = {'3': 27, '2': 28, '4': 29, '1': 30}
            idx = mapping.get(digit, None)
            if idx is None:
                continue
        elif suit == 'd':
            mapping = {'1': 31, '2': 32, '3': 33}
            idx = mapping.get(digit, None)
            if idx is None:
                continue
        else:
            continue
        vec[idx] += 1
    return vec.reshape(1, -1)

def vector_to_hand_str(vec):
    hand = ""
    for i in range(9):
        count = int(vec[i])
        hand += (str(i+1) + "m") * count
    for i in range(9, 18):
        count = int(vec[i])
        hand += (str(i-8) + "p") * count
    for i in range(18, 27):
        count = int(vec[i])
        hand += (str(i-17) + "s") * count
    wind_map = {27: "3w", 28: "2w", 29: "4w", 30: "1w"}
    for i in range(27, 31):
        count = int(vec[i])
        hand += wind_map[i] * count
    dragon_map = {31: "1d", 32: "2d", 33: "3d"}
    for i in range(31, 34):
        count = int(vec[i])
        hand += dragon_map[i] * count
    return hand

# -------------------------------
# Helper functions for partition-based grouping
# -------------------------------
def is_run(three_nums):
    return (three_nums[1] == three_nums[0] + 1) and (three_nums[2] == three_nums[1] + 1)

def is_triplet2(three_nums):
    return (three_nums[0] == three_nums[1]) and (three_nums[1] == three_nums[2])

def forms_meld(tiles_nums):
    from itertools import combinations
    if len(tiles_nums) < 3:
        return False
    for combo in combinations(tiles_nums, 3):
        if is_run(sorted(combo)) or is_triplet2(combo):
            return True
    return False

def leftover_copies(tile_num, shape):
    return max(4 - shape.count(tile_num), 0)

def acceptance_of_tiles(shape):
    total = 0
    for t in range(1, 10):
        if leftover_copies(t, shape) > 0:
            if forms_meld(shape + [t]):
                total += leftover_copies(t, shape)
    return total

def better_wait_count(shape):
    original_acc = acceptance_of_tiles(shape)
    improve_sum = 0
    for t in range(1, 10):
        if leftover_copies(t, shape) == 0:
            continue
        if forms_meld(shape + [t]):
            continue
        new_acc = acceptance_of_tiles(shape + [t])
        if new_acc > original_acc:
            improve_sum += leftover_copies(t, shape)
    return improve_sum

def acceptance_of_tiles_global(shape, global_count):
    total = 0
    for t in range(1, 10):
        available = max(4 - global_count.get(t, 0), 0)
        if available > 0:
            if forms_meld(shape + [t]):
                total += available
    return total

def better_wait_count_global(shape, global_count):
    original_acc = acceptance_of_tiles_global(shape, global_count)
    improve_sum = 0
    for t in range(1, 10):
        available = max(4 - global_count.get(t, 0), 0)
        if available == 0:
            continue
        if forms_meld(shape + [t]):
            continue
        new_acc = acceptance_of_tiles_global(shape + [t], global_count)
        if new_acc > original_acc:
            improve_sum += available
    return improve_sum

# -------------------------------
# Modified group_hand function with partition navigation capability
# -------------------------------
def group_hand(tiles):
    """
    Modified version of group_hand that normalizes weighted ukeire values
    based on the average ukeire among waiting groups in each partition.
    It now also computes an overall average normalized weighted value for the partition.
    """
    import itertools
    import math
    from collections import defaultdict

    def parse_tile_path(path):
        import os
        filename = os.path.basename(path)
        base = filename.replace(".svg", "")
        if base.startswith("Man"):
            suit = "Man"
            number = int(base[3:])
        elif base.startswith("Pin"):
            suit = "Pin"
            number = int(base[3:])
        elif base.startswith("Sou"):
            suit = "Sou"
            number = int(base[3:])
        else:
            suit = base
            number = 0
        return {"suit": suit, "number": number, "name": base, "fullPath": path}

    parsed_tiles = [parse_tile_path(p) for p in tiles]

    def tile_sort_key(t):
        suit_order_map = {
            "Man": 1,
            "Pin": 2,
            "Sou": 3,
            "Haku": 4, "Hatsu": 4, "Chun": 4,
            "Shaa": 5, "Nan": 5, "Pei": 5, "Ton": 5,
        }
        group = suit_order_map.get(t["suit"], 99)
        num_or_name = t["number"] if t["number"] != 0 else t["name"]
        return (group, num_or_name)
    parsed_tiles.sort(key=tile_sort_key)

    # Compute global counts for numeric suits
    global_counts = {}
    for suit in ["Man", "Pin", "Sou"]:
        global_counts[suit] = {n: sum(1 for tile in parsed_tiles if tile["suit"] == suit and tile["number"] == n)
                               for n in range(1, 10)}

    suit_map = defaultdict(list)
    for pt in parsed_tiles:
        suit_map[pt["suit"]].append(pt)

    # Partition numeric suit tiles
    def all_partitions_numeric(suit_tiles):
        results = []
        n = len(suit_tiles)
        def backtrack(start, current_partition):
            if start == n:
                results.append(list(current_partition))
                return
            for size in [2, 3]:
                if start + size <= n:
                    group = suit_tiles[start:start+size]
                    nums = [x["number"] for x in group]
                    if max(nums) - min(nums) <= 2:
                        backtrack(start+size, current_partition + [group])
            if start < n:
                backtrack(start+1, current_partition + [[suit_tiles[start]]])
        backtrack(0, [])
        return results

    def group_honors(honor_tiles):
        from collections import Counter
        c = Counter([h["name"] for h in honor_tiles])
        groups = []
        for tile_name, count in c.items():
            same_objs = [t for t in honor_tiles if t["name"] == tile_name]
            groups.append(same_objs)
        return [groups]

    numeric_suits = ["Man", "Pin", "Sou"]
    numeric_partitions_by_suit = {}
    for s in numeric_suits:
        numeric_partitions_by_suit[s] = all_partitions_numeric(suit_map[s])

    honor_tiles = []
    for s in suit_map:
        if s not in numeric_suits:
            honor_tiles.extend(suit_map[s])
    if len(honor_tiles) == 0:
        honor_partitions = [[]]
    else:
        honor_partitions = group_honors(honor_tiles)

    from itertools import product
    man_parts = numeric_partitions_by_suit["Man"] if "Man" in suit_map else [[]]
    pin_parts = numeric_partitions_by_suit["Pin"] if "Pin" in suit_map else [[]]
    sou_parts = numeric_partitions_by_suit["Sou"] if "Sou" in suit_map else [[]]

    all_big_combos = []
    for man_p in man_parts:
        for pin_p in pin_parts:
            for sou_p in sou_parts:
                for hon_p in honor_partitions:
                    merged = man_p + pin_p + sou_p + hon_p
                    all_big_combos.append(merged)

    # Modified group_wait_message to include normalized weighted ukeire.
    def group_wait_message(group, avg_ukeire=None):
        nums = [t["number"] for t in group if t["number"] > 0]
        if len(nums) == 0:
            return ""
        if group[0]["number"] != 0:
            suit = group[0]["suit"]
            if len(group) >= 3 and forms_meld(nums):
                return "Mentsu"
            if len(group) == 1:
                value = nums[0]
                if value in (1, 9):
                    return "Low value"
                elif value in (2, 8):
                    return "Average value"
                elif 3 <= value <= 7:
                    return "Good value"
                else:
                    return "Honor tile"
            acc = acceptance_of_tiles_global(nums, global_counts[suit])
            raw_weighted = acc + 0.3 * better_wait_count_global(nums, global_counts[suit])
            if avg_ukeire is not None and avg_ukeire != 0:
                normalized_weighted = raw_weighted / avg_ukeire
            else:
                normalized_weighted = raw_weighted
            return f"Ukeire: {acc}\nMaisu: {better_wait_count_global(nums, global_counts[suit]):.2f}"
        else:
            return ""

    def compute_shanten_for_partition(partition):
        """
        Compute shanten using the formula:
        
        8 - (2*(number of complete 3-tile groups) - (number of joints))
            - 1 (if no pair)
            - 1 (if there are exactly 6 blocks in the partition)
        
        In this version:
        - A complete 3-tile group is a group of exactly 3 tiles that forms a meld.
        - For groups that do not form a meld:
            * Two-tile groups:
                - If the two tiles are identical, add them to a pair list.
                - Otherwise, count as a joint.
            * Three-tile groups:
                - If any two tiles are identical (i.e. a pair exists), add the group to a pair list.
                - Otherwise, count the group as a complex joint.
        - Isolated (single-tile) groups are ignored.
        - After processing, if there is at least one pair (from two-tile or three-tile groups),
            designate one as a complete group (to avoid the no-pair penalty) and count any extras as joints.
        - Finally, if no joints are present at all, add an extra penalty of 2.
        """
        import itertools

        complete_groups = 0
        joints = 0  # counts joints (from two-tile non-pairs and three-tile complex joints)
        pair_list = []  # collects groups that are pairs (from 2-tile or 3-tile groups)

        for group in partition:
            if group and group[0]["number"] != 0:  # consider only numeric groups
                nums = [t["number"] for t in group if t["number"] > 0]
                if len(group) == 3:
                    sorted_nums = sorted(nums)
                    # A 3-tile meld must be exactly a triplet or a strict run
                    # (e.g., [5,6,7], not [5,6,6] or [5,7,7])
                    if (
                        (sorted_nums[0] == sorted_nums[1] == sorted_nums[2])  # triplet
                        or (sorted_nums[0] + 1 == sorted_nums[1] and sorted_nums[1] + 1 == sorted_nums[2])  # run
                    ):
                        complete_groups += 1
                    else:
                        # 3-tile shapes that don’t form a strict run or a triplet
                        # (like 566, 577, 899, etc.) are counted as joints.
                        joints += 1

                elif len(group) == 2:
                    if forms_meld(nums):
                        joints += 1
                    else:
                        if group[0]["name"] == group[1]["name"]:
                            pair_list.append(group)
                        else:
                            joints += 1
                elif len(group) == 1:
                    # Isolated tile; not considered for joints.
                    pass

        # Process collected pair groups:
        if len(pair_list) >= 1:
            # Use one pair as a complete group to avoid the "no pair" penalty.
            # Any extra pairs count as joints.
            joints += (len(pair_list) - 1)
        
        num_blocks = len(partition)
        penalty_no_pair = 0 if len(pair_list) >= 1 else 1
        penalty_six_blocks = 1 if num_blocks == 6 else 0


        shanten = 8 - 2 * complete_groups - joints + (1 if len(pair_list) == 0 else 0) 

        # If no joints are present, add a penalty of 2.
        if joints == 0:
            shanten += 2

        return shanten



    def compute_partition_scores(partition):
        total_wait = 0
        bonus_complete = 0
        pair_count = 0
        isolated_penalty = 0
        for group in partition:
            nums = [t["number"] for t in group if t["number"] > 0]
            if group and group[0]["number"] != 0:
                suit = group[0]["suit"]
                if len(group) >= 3 and forms_meld(nums):
                    bonus_complete += 1000
                elif len(group) == 2 and not forms_meld(nums):
                    total_wait += better_wait_count_global(nums, global_counts[suit])
                    if group[0]["name"] == group[1]["name"]:
                        pair_count += 1
                elif len(group) == 1:
                    isolated_penalty += 300
            else:
                pass
        bonus_pair = 500 if pair_count == 1 else 0
        num_blocks = len(partition)
        block_bonus = 0
        block_penalty = 0
        if num_blocks == 5:
            block_bonus = 2000
        elif num_blocks > 5:
            block_penalty = (num_blocks - 5) * 500
        elif num_blocks < 5:
            block_penalty = (5 - num_blocks) * 300
        total_score = total_wait + bonus_complete + bonus_pair + block_bonus - block_penalty - isolated_penalty
        return (total_wait, total_score, bonus_complete, bonus_pair, block_bonus, block_penalty, isolated_penalty)

    scored_combos = []
    for partition in all_big_combos:
        # First, compute the average raw ukeire among waiting groups
        waiting_acc_values = []
        waiting_norm_values = []
        for group in partition:
            nums = [t["number"] for t in group if t["number"] > 0]
            # Define waiting groups as groups of size 2 that do not form a meld.
            if len(group) == 2 and not forms_meld(nums):
                suit = group[0]["suit"]
                acc = acceptance_of_tiles_global(nums, global_counts[suit])
                waiting_acc_values.append(acc)
        avg_ukeire = np.mean(waiting_acc_values) if waiting_acc_values else None

        # Now, compute normalized weighted value for each waiting group and average them.
        for group in partition:
            nums = [t["number"] for t in group if t["number"] > 0]
            if len(group) == 2 and not forms_meld(nums):
                suit = group[0]["suit"]
                acc = acceptance_of_tiles_global(nums, global_counts[suit])
                raw_weighted = acc + 0.3 * better_wait_count_global(nums, global_counts[suit])
                norm_value = raw_weighted / avg_ukeire if avg_ukeire and avg_ukeire != 0 else raw_weighted
                waiting_norm_values.append(norm_value)
        avg_norm_value = np.mean(waiting_norm_values) if waiting_norm_values else 0

        partition_info = []
        for group in partition:
            message = group_wait_message(group, avg_ukeire)
            partition_info.append(([tile["fullPath"] for tile in group], message))
        overall = compute_partition_scores(partition)
        shanten_value = compute_shanten_for_partition(partition)
        # Extend overall score tuple with the computed average normalized weighted value and shanten value.
        overall_extended = overall + (avg_norm_value, shanten_value)
        scored_combos.append((partition_info, overall_extended))
    return scored_combos


# -------------------------------
# MainWindow class with partition navigation and ordering toggle
# -------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepRiichi")
        self.setGeometry(100, 100, 800, 850)
        self.setFixedWidth(1100)
        self.setStyleSheet("background-color: black;")
        self.setWindowIcon(QIcon("icon.png"))

        
        container = QWidget()
        self.layout = QVBoxLayout(container)
        
        self.title_label = QLabel("DeepRiichi")
        self.title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("SimSun", 36)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: white;")
        self.layout.addWidget(self.title_label)
        
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter hand code e.g. 1m2m3m4m5m6m7m8m1p1p1p3w3w")
        self.input_field.setStyleSheet("background-color: #222; color: white; border: 1px solid white; padding: 5px;")
        input_layout.addWidget(self.input_field)
        self.submit_button = QPushButton("Search")
        self.submit_button.setStyleSheet("background-color: #444; color: white; padding: 5px;")
        self.submit_button.clicked.connect(self.process_hand)
        input_layout.addWidget(self.submit_button)
        self.layout.addLayout(input_layout)
        
        toggle_layout = QHBoxLayout()
        self.speed_radio = QRadioButton("Speed")
        self.speed_radio.setStyleSheet("color: white;")
        self.score_radio = QRadioButton("Score")
        self.score_radio.setStyleSheet("color: white;")
        self.score_radio.setChecked(True)
        self.order_group = QButtonGroup()
        self.order_group.addButton(self.speed_radio)
        self.order_group.addButton(self.score_radio)
        toggle_layout.addWidget(self.speed_radio)
        toggle_layout.addWidget(self.score_radio)
        toggle_layout.addStretch()
        self.layout.addLayout(toggle_layout)
        self.speed_radio.toggled.connect(self.display_results)
        self.score_radio.toggled.connect(self.display_results)

        
        self.status_label = QLabel("Loading indices...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: yellow;")
        self.layout.addWidget(self.status_label)
        
        self.input_scroll_area = QScrollArea()
        self.input_scroll_area.setWidgetResizable(True)
        self.input_scroll_widget = QWidget()
        self.input_hand_layout = QHBoxLayout()
        self.input_hand_layout.setSpacing(5)
        self.input_hand_layout.setAlignment(Qt.AlignCenter)
        self.input_scroll_widget.setLayout(self.input_hand_layout)
        self.input_scroll_area.setWidget(self.input_scroll_widget)
        self.input_scroll_area.setMinimumHeight(150)
        self.layout.addWidget(self.input_scroll_area)
        
        self.results_label = QLabel("Results:")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setStyleSheet("color: white; font-size: 20px;")
        self.layout.addWidget(self.results_label)
        
        self.results_scroll_area = QScrollArea()
        self.results_scroll_area.setWidgetResizable(True)
        self.results_scroll_widget = QWidget()
        self.results_hand_layout = QVBoxLayout()
        self.results_hand_layout.setSpacing(0)
        self.results_scroll_widget.setLayout(self.results_hand_layout)
        self.results_scroll_area.setWidget(self.results_scroll_widget)
        self.results_scroll_area.setMinimumHeight(400)
        self.layout.addWidget(self.results_scroll_area)
        
        # Partition Navigation Controls frame
        self.group_buttons_frame = QFrame()
        self.group_buttons_layout = QVBoxLayout()
        self.group_buttons_layout.setSpacing(5)
        self.group_buttons_layout.setContentsMargins(5,5,5,5)
        self.group_buttons_frame.setLayout(self.group_buttons_layout)
        self.layout.addWidget(self.group_buttons_frame)
        
        self.blocks_frame = QFrame()
        self.blocks_frame.setStyleSheet("border: 2px solid lightblue;")
        self.blocks_layout = QVBoxLayout()
        self.blocks_layout.setSpacing(5)
        self.blocks_layout.setContentsMargins(5, 5, 5, 5)
        self.blocks_frame.setLayout(self.blocks_layout)
        self.layout.addWidget(self.blocks_frame)
        
        self.time_label = QLabel("")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: white;")
        self.layout.addWidget(self.time_label)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(container)
        self.setCentralWidget(scroll_area)
        
        self.show()
        QApplication.processEvents()
        self.load_model()
        self.results_list = []
        
        # Partition Navigation attributes
        self.partitions_all = []
        self.sorted_partitions = []
        self.partition_ordering = "weighted"  # "weighted" or "ukeire"
        self.current_partition_index = 0

    def load_model(self):
        self.status_label.setText("Loading indices from files...")
        QApplication.processEvents()
        self.index_full = faiss.read_index("full_index.faiss")
        self.index_full.nprobe = 10
        self.index_nonchiitoitsu = faiss.read_index("nonchiitoitsu_index.faiss")
        self.index_nonchiitoitsu.nprobe = 10
        self.data = np.load("data.npy")
        self.data_nonchiitoitsu = np.load("data_nonchiitoitsu.npy")
        self.nonchiitoitsu_indices = np.load("nonchiitoitsu_indices.npy")
        self.status_label.setText("Indices loaded. Enter a hand and press the button.")
        
    def process_hand(self):
        for layout in [self.input_hand_layout, self.results_hand_layout, self.blocks_layout, self.group_buttons_layout]:
            while layout.count():
                widget = layout.takeAt(0).widget()
                if widget:
                    widget.deleteLater()
        self.time_label.setText("")
        hand_code = self.input_field.text().strip()
        tile_paths = parse_hand(hand_code)
        if len(tile_paths) != 14:
            error_label = QLabel(f"Error: Hand must contain exactly 14 tiles (got {len(tile_paths)} tiles)")
            error_label.setStyleSheet("color: red;")
            self.input_hand_layout.addWidget(error_label)
            return
        for path in tile_paths:
            if os.path.exists(path):
                tile_frame = QFrame()
                tile_frame.setStyleSheet("background-color: white; border-radius: 10px;")
                tile_frame.setFixedSize(54, 74)
                frame_layout = QVBoxLayout(tile_frame)
                frame_layout.setContentsMargins(2, 2, 2, 2)
                svg_widget = QSvgWidget(path)
                svg_widget.setFixedSize(50, 70)
                frame_layout.addWidget(svg_widget)
                self.input_hand_layout.addWidget(tile_frame)
            else:
                error_label = QLabel("Missing: " + os.path.basename(path))
                error_label.setStyleSheet("color: red;")
                self.input_hand_layout.addWidget(error_label)
                
        # Compute and store all partitions
        self.partitions_all = group_hand(tile_paths)
        self.partition_ordering = "weighted"
        self.current_partition_index = 0
        self.update_partition_navigation()
        
        query_vec = hand_to_vector(hand_code)
        input_twos = np.sum(query_vec[0] == 2)
        candidate_k = 100
        if input_twos < 4:
            index_to_use = self.index_nonchiitoitsu
            index_mapping = self.nonchiitoitsu_indices
            print("Using non-chiitoitsu index for search.")
        else:
            index_to_use = self.index_full
            index_mapping = np.arange(self.data.shape[0])
            print("Using full index for search.")
        start_search = time.time()
        distances, indices = index_to_use.search(query_vec, candidate_k)
        end_search = time.time()
        base_search_time = end_search - start_search
        candidate_ids = index_mapping[indices[0]]
        results = []
        for i, idx in enumerate(candidate_ids):
            candidate_vec = self.data[idx]
            base_score = distances[0][i]
            pts = compute_points(candidate_vec)
            yaku_names = get_yaku_names(candidate_vec)
            results.append((idx, base_score, pts, yaku_names))
        self.results_list = results
        self.time_label.setText(f"Search completed in {base_search_time:.2f} seconds (base search).")
        self.display_results()
        
    def update_partition_navigation(self):
        # Choose sort key and order based on current ordering mode.
        if self.partition_ordering == "maisu":
            # total_wait is at index 0; higher is better.
            sort_key = lambda x: x[1][0]
            reverse = True
        else:  # self.partition_ordering == "shanten"
            # shanten_value is at index 8; lower is better.
            sort_key = lambda x: x[1][8]
            reverse = False

        self.sorted_partitions = sorted(self.partitions_all, key=sort_key, reverse=reverse)
        if self.current_partition_index >= len(self.sorted_partitions):
            self.current_partition_index = len(self.sorted_partitions) - 1
        if self.current_partition_index < 0:
            self.current_partition_index = 0

        current_partition = self.sorted_partitions[self.current_partition_index]
        # Unpack the score tuple.
        total_wait, total_score, bonus_complete, bonus_pair, block_bonus, block_penalty, isolated_penalty, avg_norm_value, shanten_value = current_partition[1]

        layout = self.group_buttons_layout
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()


        nav_layout = QHBoxLayout()
        left_button = QPushButton("<")
        left_button.setFixedWidth(50)
        left_button.setStyleSheet("background-color: #666; color: white;")
        left_button.clicked.connect(self.prev_partition)

        # Update partition summary to display shanten and maisu.
        partition_summary = QPushButton(
            f"Partition {self.current_partition_index+1} of {len(self.sorted_partitions)}\n"
            f"Shanten: {shanten_value}\nMaisu: {total_wait}"
        )
        partition_summary.setStyleSheet("background-color: #666; color: white;")
        self.show_partition_blocks(current_partition[0])
        partition_summary.clicked.connect(lambda: self.show_partition_blocks(current_partition[0]))
        nav_layout.addWidget(partition_summary)

        right_button = QPushButton(">")
        right_button.setFixedWidth(50)
        right_button.setStyleSheet("background-color: #666; color: white;")
        right_button.clicked.connect(self.next_partition)

        layout.addLayout(nav_layout)
    
    def toggle_partition_ordering(self):
        if self.partition_ordering == "maisu":
            self.partition_ordering = "shanten"
        else:
            self.partition_ordering = "maisu"
        self.current_partition_index = 0
        self.update_partition_navigation()

    
    def prev_partition(self):
        if self.current_partition_index > 0:
            self.current_partition_index -= 1
            self.update_partition_navigation()
    
    def next_partition(self):
        if self.current_partition_index < len(self.sorted_partitions) - 1:
            self.current_partition_index += 1
            self.update_partition_navigation()
    
    def show_partition_blocks(self, partition_info):
        while self.blocks_layout.count():
            widget = self.blocks_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for group in partition_info:
            paths, message = group
            group_layout = QHBoxLayout()
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(3)
            group_layout.setAlignment(Qt.AlignLeft)
            for path in paths:
                if os.path.exists(path):
                    tile_frame = QFrame()
                    tile_frame.setStyleSheet("background-color: white; border-radius: 10px;")
                    tile_frame.setFixedSize(54, 74)
                    frame_layout = QVBoxLayout(tile_frame)
                    frame_layout.setContentsMargins(0, 0, 0, 0)
                    svg_widget = QSvgWidget(path)
                    svg_widget.setFixedSize(54, 74)
                    frame_layout.addWidget(svg_widget)
                    group_layout.addWidget(tile_frame)
                else:
                    error_label = QLabel("Missing: " + os.path.basename(path))
                    error_label.setStyleSheet("color: red;")
                    group_layout.addWidget(error_label)
            msg_label = QLabel(message)
            msg_label.setStyleSheet("color: white;")
            group_layout.addWidget(msg_label)
            container = QFrame()
            container.setLayout(group_layout)
            container.setStyleSheet("border: 1px solid gray;")
            self.blocks_layout.addWidget(container)
        self.blocks_layout.addStretch()
        
    def display_results(self):
        while self.results_hand_layout.count():
            widget = self.results_hand_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        if not self.results_list:
            return
        if self.speed_radio.isChecked():
            sorted_results = sorted(self.results_list, key=lambda x: x[1])
        else:
            sorted_results = sorted(self.results_list, key=lambda x: x[2], reverse=True)
        top_results = sorted_results[:100]
        for rank, (idx, base_score, pts, yaku_names) in enumerate(top_results):
            result_vec = self.data[idx]
            hand_str = vector_to_hand_str(result_vec)
            result_tile_paths = parse_hand(hand_str)
            result_layout = QVBoxLayout()
            header_layout = QHBoxLayout()
            result_label = QLabel(f"Rank {rank+1} (Row {idx}, FAISS Score {base_score:.2f}):")
            result_label.setStyleSheet("color: white;")
            header_layout.addWidget(result_label)
            result_layout.addLayout(header_layout)
            score_label = QLabel(f"Points: {pts}  |  Yakus: {yaku_names}")
            score_label.setStyleSheet("color: lightgreen; font-size: 12pt;")
            result_layout.addWidget(score_label)
            images_layout = QHBoxLayout()
            images_layout.setContentsMargins(0, 0, 0, 0)
            images_layout.setSpacing(3)
            images_layout.setAlignment(Qt.AlignLeft)
            for path in result_tile_paths:
                if os.path.exists(path):
                    tile_frame = QFrame()
                    tile_frame.setStyleSheet("background-color: white; border-radius: 10px;")
                    tile_frame.setFixedSize(54, 74)
                    frame_layout = QVBoxLayout(tile_frame)
                    frame_layout.setContentsMargins(0, 0, 0, 0)
                    svg_widget = QSvgWidget(path)
                    svg_widget.setFixedSize(54, 74)
                    frame_layout.addWidget(svg_widget)
                    images_layout.addWidget(tile_frame)
                else:
                    error_label = QLabel("Missing: " + os.path.basename(path))
                    error_label.setStyleSheet("color: red;")
                    images_layout.addWidget(error_label)
            result_layout.addLayout(images_layout)
            container = QFrame()
            container.setLayout(result_layout)
            self.results_hand_layout.addWidget(container)

# -------------------------------
# Yaku and scoring helper functions
# -------------------------------
def triplet_penalty(vec, penalty_weight=0.7):
    return penalty_weight * np.sum(vec == 3)

def is_kokushi_musou(hand):
    indices = [0,8,9,17,18,26,27,28,29,30,31,32,33]
    pairCount = 0
    for i in indices:
        if hand[i] == 0:
            return False
        if hand[i] >= 2:
            pairCount += 1
    return (pairCount == 1)

def is_daisangen(hand):
    return (hand[31] == 3 and hand[32] == 3 and hand[33] == 3)

def is_suuankou(hand):
    nonZero = sum(1 for x in hand if x > 0)
    triplets = sum(1 for x in hand if x == 3)
    pairCount = sum(1 for x in hand if x == 2)
    return (nonZero == 5 and triplets == 4 and pairCount == 1)

def is_shousuushi(hand):
    winds = [27,28,29,30]
    triplets = sum(1 for i in winds if hand[i] == 3)
    pairCount = sum(1 for i in winds if hand[i] == 2)
    return (triplets == 3 and pairCount == 1)

def is_daisuushi(hand):
    winds = [27,28,29,30]
    return all(hand[i] >= 3 for i in winds)

def is_tsuuiisou(hand):
    if any(hand[i] != 0 for i in range(27)):
        return False
    return (sum(hand[27:34]) == 14)

def is_ryuuiisou(hand):
    allowed = [19,20,21,23,25,32]
    total = 0
    for i in range(34):
        if i not in allowed:
            if hand[i] != 0:
                return False
        else:
            total += hand[i]
    return (total == 14)

def is_chinroutou(hand):
    allowed = [0,8,9,17,18,26]
    total = 0
    for i in range(34):
        if i not in allowed:
            if hand[i] != 0:
                return False
        else:
            total += hand[i]
    return (total == 14)

def is_pinfu(hand):
    if any(x == 3 for x in hand):
        return False
    if any(hand[i] >= 2 for i in range(27,34)):
        return False
    terminals = [0,8,9,17,18,26]
    for i in terminals:
        if hand[i] == 2:
            return False
    return True

def is_iipeikou(hand):
    for s in range(3):
        start = s * 9
        for i in range(start, start+7):
            if hand[i] == 2 and hand[i+1] == 2 and hand[i+2] == 2:
                return True
    return False

def is_tanyao(hand):
    terminals = [0,8,9,17,18,26]
    if any(hand[i] > 0 for i in range(27,34)):
        return False
    return not any(hand[i] > 0 for i in terminals)

def yakuhai_points(hand):
    pts = 0
    if hand[27] == 3: pts += 1000
    if hand[31] == 3: pts += 1000
    if hand[32] == 3: pts += 1000
    if hand[33] == 3: pts += 1000
    return pts

def is_junchan(hand):
    if any(hand[i] > 0 for i in range(27, 34)):
        return False
    def can_partition(counts, need_pair):
        if sum(counts) == 0:
            return not need_pair
        if need_pair:
            if counts[0] >= 2:
                new_counts = counts.copy()
                new_counts[0] -= 2
                if can_partition(new_counts, False):
                    return True
            if counts[8] >= 2:
                new_counts = counts.copy()
                new_counts[8] -= 2
                if can_partition(new_counts, False):
                    return True
        if counts[0] > 0 and counts[1] > 0 and counts[2] > 0:
            new_counts = counts.copy()
            new_counts[0] -= 1
            new_counts[1] -= 1
            new_counts[2] -= 1
            if can_partition(new_counts, need_pair):
                return True
        if counts[6] > 0 and counts[7] > 0 and counts[8] > 0:
            new_counts = counts.copy()
            new_counts[6] -= 1
            new_counts[7] -= 1
            new_counts[8] -= 1
            if can_partition(new_counts, need_pair):
                return True
        if counts[0] >= 3:
            new_counts = counts.copy()
            new_counts[0] -= 3
            if can_partition(new_counts, need_pair):
                return True
        if counts[8] >= 3:
            new_counts = counts.copy()
            new_counts[8] -= 3
            if can_partition(new_counts, need_pair):
                return True
        return False
    def suit_counts(start):
        return [hand[i] for i in range(start, start + 9)]
    suits = {
        "Man": suit_counts(0),
        "Pin": suit_counts(9),
        "Sou": suit_counts(18)
    }
    allowed_positions = {0, 1, 2, 6, 7, 8}
    pair_needed_count = 0
    for counts in suits.values():
        for i, cnt in enumerate(counts):
            if cnt > 0 and i not in allowed_positions:
                return False
        total = sum(counts)
        if total == 0:
            continue
        if total % 3 == 0:
            need_pair = False
        elif total % 3 == 2:
            need_pair = True
            pair_needed_count += 1
        else:
            return False
        if not can_partition(counts, need_pair):
            return False
    if pair_needed_count != 1:
        return False
    return True

def is_sanshoku_doujun(hand):
    man = list(hand[0:9])
    pin = list(hand[9:18])
    sou = list(hand[18:27])
    def can_partition_suit(counts, require_pair):
        if sum(counts) == 0:
            return not require_pair
        for i in range(9):
            if counts[i] > 0:
                break
        if counts[i] >= 3:
            counts[i] -= 3
            if can_partition_suit(counts, require_pair):
                counts[i] += 3
                return True
            counts[i] += 3
        if i <= 6 and counts[i] > 0 and counts[i+1] > 0 and counts[i+2] > 0:
            counts[i] -= 1; counts[i+1] -= 1; counts[i+2] -= 1
            if can_partition_suit(counts, require_pair):
                counts[i] += 1; counts[i+1] += 1; counts[i+2] += 1
                return True
            counts[i] += 1; counts[i+1] += 1; counts[i+2] += 1
        if require_pair and counts[i] >= 2:
            counts[i] -= 2
            if can_partition_suit(counts, False):
                counts[i] += 2
                return True
            counts[i] += 2
        return False
    def suit_remove_candidate(suit_counts, k):
        if suit_counts[k] < 1 or suit_counts[k+1] < 1 or suit_counts[k+2] < 1:
            return None
        new_counts = suit_counts.copy()
        new_counts[k] -= 1; new_counts[k+1] -= 1; new_counts[k+2] -= 1
        return new_counts
    for k in range(0, 7):
        new_man = suit_remove_candidate(man, k)
        new_pin = suit_remove_candidate(pin, k)
        new_sou = suit_remove_candidate(sou, k)
        if new_man is None or new_pin is None or new_sou is None:
            continue
        valid = True
        for suit_counts in [new_man, new_pin, new_sou]:
            total = sum(suit_counts)
            require_pair = (total % 3 == 2)
            if not can_partition_suit(suit_counts.copy(), require_pair):
                valid = False
                break
        if valid:
            return True
    return False

def is_sanshoku_doukou(hand):
    for i in range(9):
        if hand[i] == 3 and hand[9+i] == 3 and hand[18+i] == 3:
            return True
    return False

def is_ikkitsuukan(hand):
    for s in range(3):
        if all(hand[s*9 + i] >= 1 for i in range(9)):
            return True
    return False

def is_chiitoitsu(hand):
    if any(x not in (0,2) for x in hand):
        return False
    return sum(1 for x in hand if x == 2) == 7

def is_honroutou(hand):
    allowed = set([0,8,9,17,18,26])
    for i in range(34):
        if i not in allowed and not (27 <= i < 34):
            if hand[i] != 0:
                return False
    return (sum(hand) == 14)

def is_honitsu(hand):
    hasMan = any(hand[i] > 0 for i in range(0,9))
    hasPin = any(hand[i] > 0 for i in range(9,18))
    hasSou = any(hand[i] > 0 for i in range(18,27))
    suitCount = int(hasMan) + int(hasPin) + int(hasSou)
    hasHonor = any(hand[i] > 0 for i in range(27,34))
    return (suitCount == 1 and hasHonor)

def is_ryanpeikou(hand):
    count = 0
    if any(x not in (0,2) for x in hand):
        return False
    for s in range(3):
        start = s * 9
        for i in range(start, start+7):
            if hand[i] == 2 and hand[i+1] == 2 and hand[i+2] == 2:
                count += 1
    return count >= 2

def is_chinitsu(hand):
    hasMan = any(hand[i] > 0 for i in range(0,9))
    hasPin = any(hand[i] > 0 for i in range(9,18))
    hasSou = any(hand[i] > 0 for i in range(18,27))
    if any(hand[i] > 0 for i in range(27,34)):
        return False
    suitCount = int(hasMan) + int(hasPin) + int(hasSou)
    return suitCount == 1

def is_chanta(hand):
    def allowed_pair(i):
        if i < 27:
            mod = i % 9
            return mod == 0 or mod == 8
        else:
            return True
    def allowed_triplet(i):
        if i < 27:
            mod = i % 9
            return mod == 0 or mod == 8
        else:
            return True
    def allowed_sequence(i):
        mod = i % 9
        return mod == 0 or mod == 6
    def helper(hand, pair_used):
        if sum(hand) == 0:
            return True
        for i in range(34):
            if hand[i] > 0:
                break
        if not pair_used and hand[i] >= 2 and allowed_pair(i):
            hand[i] -= 2
            if helper(hand, True):
                hand[i] += 2
                return True
            hand[i] += 2
        if i < 27:
            if hand[i] >= 3 and allowed_triplet(i):
                hand[i] -= 3
                if helper(hand, pair_used):
                    hand[i] += 3
                    return True
                hand[i] += 3
            mod = i % 9
            if allowed_sequence(i) and mod <= 6:
                if hand[i] > 0 and hand[i+1] > 0 and hand[i+2] > 0:
                    hand[i] -= 1
                    hand[i+1] -= 1
                    hand[i+2] -= 1
                    if helper(hand, pair_used):
                        hand[i] += 1
                        hand[i+1] += 1
                        hand[i+2] += 1
                        return True
                    hand[i] += 1
                    hand[i+1] += 1
                    hand[i+2] += 1
        else:
            if hand[i] >= 3:
                hand[i] -= 3
                if helper(hand, pair_used):
                    hand[i] += 3
                    return True
                hand[i] += 3
        return False
    hand_copy = list(hand)
    return helper(hand_copy, False)

def compute_points(result_vec):
    hand = result_vec.astype(int).flatten()
    if is_kokushi_musou(hand): return 48000
    if is_daisangen(hand): return 48000
    if is_suuankou(hand): return 48000
    if is_shousuushi(hand): return 48000
    if is_daisuushi(hand): return 48000
    if is_tsuuiisou(hand): return 48000
    if is_ryuuiisou(hand): return 48000
    if is_chinroutou(hand): return 48000
    pts = 1000
    if is_pinfu(hand): pts += 1000
    if is_iipeikou(hand): pts += 1000
    if is_tanyao(hand): pts += 1000
    pts += yakuhai_points(hand)
    if is_chanta(hand): pts += 2000
    if is_junchan(hand): pts += 3000
    if is_sanshoku_doujun(hand): pts += 2000
    if is_sanshoku_doukou(hand): pts += 3000
    if is_ikkitsuukan(hand): pts += 2000
    if is_chiitoitsu(hand): pts += 2000
    if is_honitsu(hand): pts += 3000
    if is_ryanpeikou(hand): pts += 3000
    if is_chinitsu(hand): pts += 6000
    return pts

def get_yaku_names(result_vec):
    hand = result_vec.astype(int).flatten()
    names = []
    if is_kokushi_musou(hand):
        names.append("KokushiMusou")
    elif is_daisangen(hand):
        names.append("Daisangen")
    elif is_suuankou(hand):
        names.append("Suuankou")
    elif is_shousuushi(hand):
        names.append("Shousuushi")
    elif is_daisuushi(hand):
        names.append("Daisuushi")
    elif is_tsuuiisou(hand):
        names.append("Tsuuiisou")
    elif is_ryuuiisou(hand):
        names.append("Ryuuiisou")
    elif is_chinroutou(hand):
        names.append("Chinroutou")
    else:
        names.append("Riichi")
        if is_chiitoitsu(hand):
            names.append("Chiitoitsu")
        if is_pinfu(hand):
            names.append("Pinfu")
        if is_tanyao(hand):
            names.append("Tanyao")
        if yakuhai_points(hand) > 0:
            names.append("Yakuhai")
        if is_junchan(hand):
            names.append("Junchan")
        else:
            if is_chanta(hand):
                names.append("Chanta")
        if is_sanshoku_doujun(hand):
            names.append("SanshokuDoujun")
        if is_sanshoku_doukou(hand):
            names.append("SanshokuDoukou")
        if is_ikkitsuukan(hand):
            names.append("Ikkitsuukan")
        if is_honitsu(hand):
            names.append("Honitsu")
        if is_ryanpeikou(hand):
            names.append("Ryanpeikou")
        else:
            if is_iipeikou(hand):
                names.append("Iipeikou")
        if is_chinitsu(hand):
            names.append("Chinitsu")
    return " ".join(names)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
