class Node(object):
        def __init__(self, x, y):
            self.prev = None
            self.nxt = None
            self.x = x
            self.y = y

        def set_prev(self, prev):
            self.prev = prev

        def set_next(self,nxt):
            self.nxt = nxt