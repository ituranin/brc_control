import time

class StopWatch:
    def __init__(self):
        self.startTime = 0
        self.stopTime = 0
        self.finalTime = 0

    def start(self):
        self.startTime = int(round(time.time() * 1000))
        
    def stop(self):
        self.stopTime = int(round(time.time() * 1000))
        self.finalTime = self.stopTime - self.startTime
        return self.finalTime
    
    def getSeconds(self):
        return self.finalTime / 1000