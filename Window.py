class Window():
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
    
    def size(self, iteration):
        i = 0
        while (iteration >= self.sizes[i][0] and i < (len(self.sizes) - 1)):
            i += 1
        return self.sizes[i][1]