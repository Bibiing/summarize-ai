import json
from pathlib import Path

class JSONLogger:
    def __init__(self, log_file="logs/apps.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logs = []

        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    existing_runs = json.load(f)
                self.run_id = len(existing_runs) + 1
            except Exception:
                self.run_id = 1  
        else:
            self.run_id = 1
        
    def log(self, step, status, message, **kwargs):
        log_entry = {
            "step": step,
            "status": status, 
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
        
        # Also print to console for immediate feedback
        status_emoji = {"SUCCESS": "✅", "ERROR": "❌", "INFO": "ℹ️", "WARNING": "⚠️"}
        print(f"{status_emoji.get(status, '📝')} [{step}] {message}")

    def save(self): 
        current_run = {
            "run_id": self.run_id,
            "logs": self.logs
        }
        
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    existing_runs = json.load(f)
                if isinstance(existing_runs, list):
                    existing_runs.append(current_run)
                else:
                    existing_runs = [existing_runs, current_run]
            except Exception:
                existing_runs = [current_run]
        else:
            existing_runs = [current_run]

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_runs, f, ensure_ascii=False, indent=2)