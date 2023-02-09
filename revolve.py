import numpy as np
from stl import mesh
import math
import sys
import ast

axis_cos = 0
axis_sin = 1
axis_height = 2

vert_type = np.float64
index_type = np.int32

def create_mesh(verts, faces):
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, face in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = verts[face[j], :]
    
    return m

def merge_meshes(meshes):
    return mesh.Mesh(np.concatenate([m.data for m in meshes]))

def create_circle(axes, radius, segments):
    verts = np.zeros((segments+1, 3), vert_type)
    faces = np.zeros((segments, 3), index_type)
    verts[0] = [0, 0, 0]

    delta_angle = (math.pi * 2) / segments

    for i in range(segments):
        angle = delta_angle * i
        vert = [0, 0, 0]
        vert[axes[axis_cos]] = math.cos(angle) * radius
        vert[axes[axis_sin]] = math.sin(angle) * radius
        verts[i+1] = vert

        j = 1 if i == segments-1 else i + 2
        faces[i] = [0, i+1, j]

    return create_mesh(verts, faces)

def create_bands(axes, radii, segment_height, segment_count):
    radius_count = len(radii)
    band_count = radius_count - 1

    verts = np.zeros((segment_count*radius_count, 3), vert_type)
    faces = np.zeros((segment_count*band_count*2, 3), index_type)
    
    delta_angle = (math.pi * 2) / segment_count

    vert_indices = np.zeros((radius_count, segment_count))

    vert_index = 0
    for radius_index, radius in enumerate(radii):
        for segment_index in range(segment_count):
            angle = delta_angle * segment_index

            vert = [0, 0, 0]
            vert[axes[axis_cos]] = math.cos(angle) * radius
            vert[axes[axis_sin]] = math.sin(angle) * radius
            vert[axes[axis_height]] = radius_index * segment_height

            verts[vert_index] = vert
            vert_indices[radius_index, segment_index] = vert_index
            vert_index += 1

    face_index = 0
    for band_index in range(band_count):
        for segment_index in range(segment_count):
            # a---b
            # | \ |
            # c---d
            a = vert_indices[band_index, segment_index]
            b = vert_indices[band_index, (segment_index+1)%segment_count]
            c = vert_indices[band_index+1, segment_index]
            d = vert_indices[band_index+1, (segment_index+1)%segment_count]
            faces[face_index] = [a, b, d]
            faces[face_index+1] = [a, d, c]
            face_index += 2
            
    
    return create_mesh(verts, faces)

def create_bands_with_caps(axes, radii, segment_height, segment_count):
    radius_count = len(radii)
    band_count = radius_count - 1

    verts = np.zeros((segment_count*radius_count+2, 3), vert_type)
    faces = np.zeros((segment_count*band_count*2+segment_count*2, 3), index_type)
    
    delta_angle = (math.pi * 2) / segment_count

    cap_centers_indices = np.zeros(2)
    vert_indices = np.zeros((radius_count, segment_count))

    cap_heights = [0, segment_height*band_count]

    vert_index = 0
    for cap_index in range(2):

        cap_height = cap_heights[cap_index]

        vert = [0, 0, 0]
        vert[axes[axis_height]] = cap_height
        verts[vert_index] = vert
        cap_centers_indices[cap_index] = vert_index
        vert_index += 1

    for radius_index, radius in enumerate(radii):
        for segment_index in range(segment_count):
            angle = delta_angle * segment_index

            vert = [0, 0, 0]
            vert[axes[axis_cos]] = math.cos(angle) * radius
            vert[axes[axis_sin]] = math.sin(angle) * radius
            vert[axes[axis_height]] = radius_index * segment_height

            verts[vert_index] = vert
            vert_indices[radius_index, segment_index] = vert_index
            vert_index += 1

    face_index = 0
    for band_index in range(band_count):
        for segment_index in range(segment_count):
            # a---b
            # | \ |
            # c---d
            a = vert_indices[band_index, segment_index]
            b = vert_indices[band_index, (segment_index+1)%segment_count]
            c = vert_indices[band_index+1, segment_index]
            d = vert_indices[band_index+1, (segment_index+1)%segment_count]
            faces[face_index] = [a, b, d]
            faces[face_index+1] = [a, d, c]
            face_index += 2
    

    bottom = cap_centers_indices[0]
    top = cap_centers_indices[1]
    for segment_index in range(segment_count):
        # bottom
        #  / \
        # a---b

        # c---d
        #  \ /
        #  top
        a = vert_indices[0, segment_index]
        b = vert_indices[0, (segment_index+1)%segment_count]
        c = vert_indices[band_count, segment_index]
        d = vert_indices[band_count, (segment_index+1)%segment_count]
        faces[face_index] = [bottom, b, a]
        faces[face_index+1] = [top, c, d]
        face_index += 2
            
    
    return create_mesh(verts, faces)


def create_revolution(axes, radius_func, start_height, end_height, band_count, segment_count):
    height = end_height - start_height
    segment_height = height / band_count
    inputs = np.linspace(start_height, end_height, band_count+1)
    radii = list(map(radius_func, inputs))
    return create_bands(axes, radii, segment_height, segment_count)

def create_capped_revolution(axes, radius_func, start_height, end_height, band_count, segment_count):
    height = end_height - start_height
    segment_height = height / band_count
    inputs = np.linspace(start_height, end_height, band_count+1)
    radii = list(map(radius_func, inputs))
    return create_bands_with_caps(axes, radii, segment_height, segment_count)

def create_shell(axes, height, radius_inner, radius_outer, segment_count):

    verts = np.zeros((segment_count*4, 3), vert_type)
    faces = np.zeros((segment_count*4*2, 3), index_type)

    vert_indices = np.zeros((4, segment_count))
    
    # 0-1
    # | |
    # 3-2

    ring_radii = [radius_outer, radius_inner, radius_inner, radius_outer]
    ring_heights = [height, height, 0, 0]

    delta_angle = (math.pi * 2) / segment_count

    vert_index = 0
    for ring_index in range(4):

        ring_radius = ring_radii[ring_index]
        ring_height = ring_heights[ring_index]

        for segment_index in range(segment_count):
            angle = delta_angle * segment_index

            vert = [0, 0, 0]
            vert[axes[axis_cos]] = math.cos(angle) * ring_radius
            vert[axes[axis_sin]] = math.sin(angle) * ring_radius
            vert[axes[axis_height]] = ring_height

            verts[vert_index] = vert
            vert_indices[ring_index, segment_index] = vert_index
            vert_index += 1

    face_index = 0
    for surface_index in range(4):
        for segment_index in range(segment_count):
            # a---b
            # | \ |
            # c---d
            a = vert_indices[surface_index, segment_index]
            b = vert_indices[surface_index, (segment_index+1)%segment_count]
            c = vert_indices[(surface_index+1)%4, segment_index]
            d = vert_indices[(surface_index+1)%4, (segment_index+1)%segment_count]
            faces[face_index] = [a, b, d]
            faces[face_index+1] = [a, d, c]
            face_index += 2
    
    return create_mesh(verts, faces)


def create_cylinder(axes, height, radius, segment_count):

    verts = np.zeros((segment_count*2+2, 3), vert_type)
    faces = np.zeros((segment_count*4*2, 3), index_type)

    centers_indices = np.zeros(2)
    ring_vert_indices = np.zeros((2, segment_count))

    ring_heights = [0, height]

    delta_angle = (math.pi * 2) / segment_count

    vert_index = 0
    for ring_index in range(2):

        ring_height = ring_heights[ring_index]

        vert = [0, 0, 0]
        vert[axes[axis_height]] = ring_height
        verts[vert_index] = vert
        centers_indices[ring_index] = vert_index
        vert_index += 1

        for segment_index in range(segment_count):
            angle = delta_angle * segment_index

            vert = [0, 0, 0]
            vert[axes[axis_cos]] = math.cos(angle) * radius
            vert[axes[axis_sin]] = math.sin(angle) * radius
            vert[axes[axis_height]] = ring_height

            verts[vert_index] = vert
            ring_vert_indices[ring_index, segment_index] = vert_index
            vert_index += 1

    face_index = 0
    bottom = centers_indices[0]
    top = centers_indices[1]
    for segment_index in range(segment_count):
        # bottom
        #  / \
        # a---b
        # | \ |
        # c---d
        #  \ /
        #  top
        a = ring_vert_indices[0, segment_index]
        b = ring_vert_indices[0, (segment_index+1)%segment_count]
        c = ring_vert_indices[1, segment_index]
        d = ring_vert_indices[1, (segment_index+1)%segment_count]
        faces[face_index] = [a, b, d]
        faces[face_index+1] = [a, d, c]
        faces[face_index+2] = [bottom, b, a]
        faces[face_index+3] = [top, c, d]
        face_index += 4
    
    return create_mesh(verts, faces)

def create_shells(axes, height_func, start_radius, end_radius, shell_count, segment_count, gap):

    shells = []

    radii = np.linspace(start_radius, end_radius, shell_count+1)

    for shell_index in range(shell_count):

        radius_inner = radii[shell_index]
        radius_outer = radii[shell_index+1] - gap
        
        height_inner = height_func(radius_inner)
        height_outer = height_func(radius_outer)
        height = min(height_inner, height_outer)

        if height != 0:
            if radius_inner == 0:
                shells.append(create_cylinder(axes, height, radius_outer, segment_count))
            else:
                shells.append(create_shell(axes, height, radius_inner, radius_outer, segment_count))
    
    return merge_meshes(shells)


axis_x = 0
axis_y = 1
axis_z = 2

#          cos     sin     height
along_y = (axis_z, axis_x, axis_y)
along_z = (axis_x, axis_y, axis_z)
along_x = (axis_y, axis_z, axis_x)

# create_circle((2, 0), 10, 32).save("circle.stl")

# create_bands(along_y, (10, 8, 4), 3, 32).save("band.stl")

# create_revolution(along_y, lambda y: math.sqrt(10-y), 0, 5, 8, 32).save("paraboloid-y.stl")

# create_revolution(along_z, lambda y: math.sqrt(10-y), 0, 10, 8, 32).save("paraboloid-z.stl")

# create_shell(along_x, 10, 3, 4, 32).save("shell-x.stl")
# create_shell(along_y, 10, 3, 4, 32).save("shell-y.stl")
# create_shell(along_z, 10, 3, 4, 32).save("shell-z.stl")

# create_cylinder(along_y, 10, 5, 32).save("cylinder-y.stl")

# create_shells(along_z, lambda r: 10-r*r, 0, math.sqrt(10), 8, 32, 0.1).save("paraboloid-shells-z.stl")

# create_shells(along_z, lambda r: (math.cos(r*.5*math.pi)+2)*(10-math.fabs(r)), 0, 10, 20, 32, 0.01).save("squiggles-shells-z.stl")

# create_capped_revolution(along_y, lambda y: math.sqrt(10-y), 0, 10, 8, 32).save("paraboloid-capped-y.stl")

# quit()


### from https://stackoverflow.com/questions/2371436/evaluating-a-mathematical-expression-in-a-string because im lazy
math_locals = {key: value for (key,value) in vars(math).items() if key[0] != '_'}
math_locals.update({"abs": abs, "min": min, "max": max, "pow": pow, "round": round})

class Visitor(ast.NodeVisitor):
    def visit(self, node):
       if not isinstance(node, self.whitelist):
           raise ValueError(node)
       return super().visit(node)

    whitelist = (ast.Module, ast.Expr, ast.Load, ast.Expression, ast.Add, ast.Sub, ast.UnaryOp, ast.Num, ast.BinOp,
            ast.Mult, ast.Div, ast.Pow, ast.BitOr, ast.BitAnd, ast.BitXor, ast.USub, ast.UAdd, ast.FloorDiv, ast.Mod,
            ast.LShift, ast.RShift, ast.Invert, ast.Call, ast.Name)

def math_evaluate(expr, locals = {}):
    locals.update(math_locals)
    if any(elem in expr for elem in '\n#') : raise ValueError(expr)
    try:
        node = ast.parse(expr.strip(), mode='eval')
        Visitor().visit(node)
        return eval(compile(node, "<string>", "eval"), {'__builtins__': None}, locals)
    except Exception: raise ValueError(expr)

###


class Argument:
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = prompt
    
    def invalid_reason(self, text):
        return None if len(text) != 0 else "Cannot be blank"
    
    def get_value(self, text):
        return text
    
    def enabled(self):
        return True
    
class ArgumentEnum(Argument):
    def __init__(self, name, prompt, options):
        super().__init__(name, prompt)
        self.options = options

    def invalid_reason(self, text):
        if text in self.options.keys():
            return None
        else:
            return "Must be one of the following: {0}".format(", ".join(self.options))
    
    def get_value(self, text):
        return self.options[text]

class ArgumentInteger(Argument):
    def __init__(self, name, prompt, invalid_reason_func = lambda value: None):
        super().__init__(name, prompt)
        self.invalid_reason_func = invalid_reason_func
    
    def invalid_reason(self, text):
        try:
            result = math_evaluate(text)
            result_type = type(result)
            if result_type != int:
                return "Must be an integer"
            else:
                return self.invalid_reason_func(result)
        except Exception as e:
            return "Failed to evaluate, error: {0}".format(e)

    def get_value(self, text):
        return math_evaluate(text)

class ArgumentNumber(Argument):
    def __init__(self, name, prompt, invalid_reason_func = lambda value: None):
        super().__init__(name, prompt)
        self.invalid_reason_func = invalid_reason_func
    
    def invalid_reason(self, text):
        try:
            result = math_evaluate(text)
            result_type = type(result)
            if result_type != int and result_type != float:
                return "Must be a number"
            else:
                return self.invalid_reason_func(result)
        except Exception as e:
            return "Failed to evaluate, error: {0}".format(e)

    def get_value(self, text):
        return math_evaluate(text)
        

class ArgumentFunction(Argument):
    def __init__(self, name, prompt, test_funcs, input_name):
        super().__init__(name, prompt)
        self.test_funcs = test_funcs
        self.input_name = input_name

    def evaluate(self, text, variables):
        return math_evaluate(text, variables)
    
    def invalid_reason(self, text):
        try:
            result = self.evaluate(text, {key: value() for key, value in self.test_funcs.items()})
            result_type = type(result)
            if result_type == int or result_type == float:
                return None
            else:
                return "Function did not return a number: \"{0}\"".format(result)
        except Exception as e:
            return "Failed to evaluate, error: {0}".format(e)
    
    def get_value(self, text):
        return lambda input_variable: self.evaluate(text, {self.input_name: input_variable})

class ArgumentOptional(Argument):
    def __init__(self, enabled_func, sub_arg):
        super().__init__(sub_arg.name, sub_arg.prompt)
        self.enabled_func = enabled_func
        self.sub_arg = sub_arg
    
    def invalid_reason(self, text):
        return self.sub_arg.invalid_reason(text)
    
    def get_value(self, text):
        return self.sub_arg.get_value(text)
    
    def enabled(self):
        return self.enabled_func()

def larger_than(maximum_getter):
    return lambda value: None if value > maximum_getter() else "Must be larger than {0}".format(maximum_getter())

saved_arg_values = {}

arguments = [
    Argument("filename", "output file name (will append .stl): "),
    ArgumentEnum("model_type", "type (shells, smooth): ", {"shells": "shells", "smooth": "smooth"}),
    ArgumentEnum("axis", "axis (x, y, z): ", {"x": along_x, "y": along_y, "z": along_z}),

    ArgumentOptional(lambda: saved_arg_values["model_type"] == "shells", ArgumentNumber("radius_start", "radius start (example \"0\"): ")),
    ArgumentOptional(lambda: saved_arg_values["model_type"] == "shells", ArgumentNumber("radius_end", "radius end (example \"sqrt(10)\"): ", larger_than(lambda: saved_arg_values["radius_start"]))),
    ArgumentOptional(lambda: saved_arg_values["model_type"] == "shells", ArgumentFunction("height_func", "height as a function of radius (r) (example \"10-r**2\"): ", {"r": lambda: saved_arg_values["radius_start"]}, "r")),
    ArgumentOptional(lambda: saved_arg_values["model_type"] == "shells", ArgumentInteger("shell_count", "shell count (example \"8\"): ", larger_than(lambda: 0))),

    ArgumentOptional(lambda: saved_arg_values["model_type"] == "smooth", ArgumentNumber("height_start", "height start (example \"0\"): ")),
    ArgumentOptional(lambda: saved_arg_values["model_type"] == "smooth", ArgumentNumber("height_end", "height end (example \"10\"): ", larger_than(lambda: saved_arg_values["height_start"]))),
    ArgumentOptional(lambda: saved_arg_values["model_type"] == "smooth", ArgumentFunction("radius_func", "radius as a function of height (y) (example: \"sqrt(10-y)\"): ", {"y": lambda: saved_arg_values["height_start"]}, "y")),
    ArgumentOptional(lambda: saved_arg_values["model_type"] == "smooth", ArgumentInteger("detail_count", "vertial detail (example \"16\"): ", larger_than(lambda: 0))),

    ArgumentInteger("segment_count", "circle detail (example \"32\"): ", larger_than(lambda: 1)),
    ArgumentOptional(lambda: saved_arg_values["model_type"] == "shells", ArgumentNumber("gap", "gap size (example \"0.1\"): ", larger_than(lambda: 0))),
]

def prompt_user(inputs):
    i = 0
    for arg in arguments:
        if not arg.enabled():
            continue
        text = None
        if i < len(inputs):
            text = inputs[i]
            invalid_reason = arg.invalid_reason(text)
            print("{0}{1}".format(arg.prompt, text))
            if invalid_reason != None:
                print("^ {}".format(invalid_reason))
                quit()

        else:
            invalid_reason = ""
            while invalid_reason != None:
                text = input(arg.prompt)
                invalid_reason = arg.invalid_reason(text)
                if invalid_reason != None:
                    print(invalid_reason)
                    continue
        i += 1
        saved_arg_values[arg.name] = arg.get_value(text)


prompt_user(sys.argv[1:])
model_type = saved_arg_values["model_type"]
axis = saved_arg_values["axis"]
segment_count = saved_arg_values["segment_count"]
filename = saved_arg_values["filename"]

if model_type == "shells":
    radius_start = saved_arg_values["radius_start"]
    radius_end = saved_arg_values["radius_end"]
    height_func = saved_arg_values["height_func"]
    gap = saved_arg_values["gap"]
    shell_count = saved_arg_values["shell_count"]
    create_shells(axis, height_func, radius_start, radius_end, shell_count, segment_count, gap).save("{0}.stl".format(filename))

elif model_type == "smooth":
    height_start = saved_arg_values["height_start"]
    height_end = saved_arg_values["height_end"]
    radius_func = saved_arg_values["radius_func"]
    detail_count = saved_arg_values["detail_count"]
    create_capped_revolution(axis, radius_func, height_start, height_end, detail_count, segment_count).save("{0}.stl".format(filename))

else:
    print("Unimplemented model type: " + model_type)
