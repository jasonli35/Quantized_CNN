class SaveOutput:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in):
        self.outputs.append(module_in)
    def clear(self):
        self.outputs = []
        
    @staticmethod
    def hook(a_layer, hook_type, save_output):
        hook_blocks = []
        for layer in a_layer:
            if isinstance(layer, hook_type):
                print("prehooked")
                hook_blocks.append(layer)
                layer.register_forward_pre_hook(save_output)       ## Input for the module will be grapped   
        return hook_blocks
        
######### Save inputs from selected layer ##########
# save_output = SaveOutput()

