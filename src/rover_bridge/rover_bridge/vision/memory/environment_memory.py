"""
EnvironmentMemory — Phase 2 module. Stub ready for Phase 1.
Stores room knowledge: what rooms exist, what connects to what,
captured images, GPT descriptions, and door positions.

In Phase 1: this file exists but is never called.
In Phase 2: import and use in main.py after room scanning logic.
"""
import json
import os
from datetime import datetime
from rover_bridge.vision.utils.logger import RoverLogger


class EnvironmentMemory:
    def __init__(self, settings):
        self.settings  = settings
        self.logger    = RoverLogger("EnvironmentMemory")
        self.file_path = settings.MEMORY_FILE
        self.data      = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                self.logger.info(f"Memory loaded from {self.file_path}")
                return json.load(f)
        return {
            "rooms": {},
            "current_location": None,
            "exploration_history": []
        }

    def save(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=2)
        self.logger.info("Memory saved.")

    def add_room(self, room_name: str, description: str, images: list[str], confirmed_by: str = "gpt"):
        self.data["rooms"][room_name] = {
            "confirmed":       True,
            "confirmed_by":    confirmed_by,
            "description":     description,
            "images":          images,
            "adjacent_rooms":  {},
            "door_positions":  {},
            "added_at":        datetime.now().isoformat(),
        }
        self.save()

    def link_rooms(self, from_room: str, direction: str, to_room: str, door_width: float = None):
        """
        Record that exiting from_room via direction leads to to_room.
        Also records the reverse link.
        """
        if from_room in self.data["rooms"]:
            self.data["rooms"][from_room]["adjacent_rooms"][direction] = {
                "leads_to":         to_room,
                "door_width_meters": door_width,
            }
        # Reverse
        opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}.get(direction)
        if opposite and to_room in self.data["rooms"]:
            self.data["rooms"][to_room]["adjacent_rooms"][opposite] = {
                "leads_to":         from_room,
                "door_width_meters": door_width,
            }
        self.save()

    def set_current_location(self, room_name: str):
        self.data["current_location"] = room_name
        if room_name not in self.data["exploration_history"]:
            self.data["exploration_history"].append(room_name)
        self.save()

    def get_current_location(self) -> str | None:
        return self.data.get("current_location")

    def knows_room(self, room_name: str) -> bool:
        return room_name in self.data["rooms"]

    def path_to(self, target_room: str) -> list[str] | None:
        """
        Simple BFS to find route from current location to target.
        Returns ordered list of rooms to traverse.
        Phase 2 use: "go to kitchen"
        """
        start = self.get_current_location()
        if not start or not self.knows_room(target_room):
            return None

        from collections import deque
        queue   = deque([[start]])
        visited = {start}

        while queue:
            path = queue.popleft()
            node = path[-1]
            if node == target_room:
                return path
            for direction, info in self.data["rooms"].get(node, {}).get("adjacent_rooms", {}).items():
                neighbor = info.get("leads_to")
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None
