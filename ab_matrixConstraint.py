import math
import maya.cmds as cmds
import maya.api.OpenMaya as apiOM


def maya_useNewAPI():
    """
    tells maya the plugin uses Maya Python API 2.0
    """
    pass


class MatrixConstraintNode(apiOM.MPxNode):

    plugin_name = "matrixConstraint"

    # plugin ID is placeholder for testing and developing purpose
    # not recommended to use it
    plugin_id = apiOM.MTypeId(0x00000000)

    # inputs
    input_use_offset_attr = apiOM.MObject()
    input_offset_matrix_attr = apiOM.MObject()

    input_compound_attr = apiOM.MObject()
    input_driver_matrix_attr = apiOM.MObject()
    input_driver_float_attr = apiOM.MObject()

    input_parent_inverse_matrix_attr = apiOM.MObject()

    # outputs
    output_translate_attr = apiOM.MObject()
    output_rotate_attr = apiOM.MObject()

    def __init__(self):
        apiOM.MPxNode.__init__(self)

        self.og_driven = None
        self.og_wm = None

        self.og_pos = list()
        self.og_rot = list()

        self.offset = None

        self.driver_names = list()
        self.driver_offsets = list()

    @staticmethod
    def creator():
        return MatrixConstraintNode()

    @staticmethod
    def initialize():

        """
        called during initialization.
        create, add and set preference for attrs and set their dependencies to each other

        :return: None
        """

        # create attr types
        compound_attr_type = apiOM.MFnCompoundAttribute()
        matrix_attr_type = apiOM.MFnMatrixAttribute()
        numerical_attr_type = apiOM.MFnNumericAttribute()

        # create input attrs
        MatrixConstraintNode.input_use_offset_attr = numerical_attr_type.create("useOffset", "uo",
                                                                                apiOM.MFnNumericData.kBoolean, True)
        MatrixConstraintNode.input_offset_matrix_attr = matrix_attr_type.create("offset", "o")

        MatrixConstraintNode.input_parent_inverse_matrix_attr = matrix_attr_type.create("parentInverseMatrix", "pim")
        matrix_attr_type.hidden = True

        MatrixConstraintNode.input_driver_matrix_attr = matrix_attr_type.create("matrix", "m")
        matrix_attr_type.disconnectBehavior = matrix_attr_type.kDelete
        MatrixConstraintNode.input_driver_float_attr = numerical_attr_type.create("weight", "w",
                                                                                  apiOM.MFnNumericData.kFloat, 1.0)
        numerical_attr_type.setMin(0.0)
        numerical_attr_type.setMax(1.0)

        MatrixConstraintNode.input_compound_attr = compound_attr_type.create("matrixIn", "mi")
        compound_attr_type.array = True
        compound_attr_type.addChild(MatrixConstraintNode.input_driver_matrix_attr)
        compound_attr_type.addChild(MatrixConstraintNode.input_driver_float_attr)

        # create output attrs
        MatrixConstraintNode.output_translate_attr = numerical_attr_type.createPoint("translate", "t")
        numerical_attr_type.writable = False
        MatrixConstraintNode.output_rotate_attr = numerical_attr_type.createPoint("rotate", "r")
        numerical_attr_type.writable = False

        # add attrs to the node
        MatrixConstraintNode.addAttribute(MatrixConstraintNode.input_use_offset_attr)
        MatrixConstraintNode.addAttribute(MatrixConstraintNode.input_offset_matrix_attr)
        MatrixConstraintNode.addAttribute(MatrixConstraintNode.input_compound_attr)
        MatrixConstraintNode.addAttribute(MatrixConstraintNode.input_parent_inverse_matrix_attr)

        MatrixConstraintNode.addAttribute(MatrixConstraintNode.output_translate_attr)
        MatrixConstraintNode.addAttribute(MatrixConstraintNode.output_rotate_attr)

        # assign which attr affects which attr
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_parent_inverse_matrix_attr,
                                              MatrixConstraintNode.output_translate_attr)
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_parent_inverse_matrix_attr,
                                              MatrixConstraintNode.output_rotate_attr)
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_use_offset_attr,
                                              MatrixConstraintNode.output_translate_attr)
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_use_offset_attr,
                                              MatrixConstraintNode.output_rotate_attr)
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_offset_matrix_attr,
                                              MatrixConstraintNode.output_translate_attr)
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_offset_matrix_attr,
                                              MatrixConstraintNode.output_rotate_attr)
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_compound_attr,
                                              MatrixConstraintNode.output_translate_attr)
        MatrixConstraintNode.attributeAffects(MatrixConstraintNode.input_compound_attr,
                                              MatrixConstraintNode.output_rotate_attr)

    def postConstructor(self):

        """
        will get called immediately after the constructor when it is safe to call any MPxNode member function
        """

        # tell node to not compute without inputs and outputs connected
        MatrixConstraintNode.setExistWithoutInConnections = True
        MatrixConstraintNode.setExistWithoutOutConnections = False

        # attach a callback function to node (gets called just before deletion)
        apiOM.MNodeMessage.addNodeAboutToDeleteCallback(self.thisMObject(), MatrixConstraintNode.on_delete)

    def connectionMade(self, plug, otherPlug, asSrc):

        """
        called when connections are made to attributes of this node.

        :param plug: MPlug - attribute on this node.
        :param otherPlug: MPlug - attribute on other node.
        :param asSrc: bool - is this plug a source of the connection.
        :return: None
        """

        # get name, translate, rotate and world matrix of first connected driven obj
        if asSrc and self.og_driven is None:

            self.og_driven = cmds.listConnections(apiOM.MFnDependencyNode(plug.node()).name(), skipConversionNodes=True, source=False)[0]
            self.og_wm = apiOM.MMatrix(cmds.getAttr("{}.worldMatrix".format(self.og_driven)))

            tform_matrix = apiOM.MTransformationMatrix(self.og_wm)
            self.og_pos = tform_matrix.translation(apiOM.MSpace.kWorld)
            self.og_rot = tform_matrix.rotation()

            for index, radian in enumerate(list(self.og_rot)):
                self.og_rot[index] = math.degrees(radian)

    def compute(self, plug, data):

        """
        recompute the given output based on the nodes inputs.

        :param plug: MPlug - plug representing the attribute that needs to be recomputed.
        :param data: MDataBlock - data block containing storage for the node's attributes.
        :return: None
        """

        # reference to the node itself
        node = self.thisMObject()
        # store node name
        node_name = apiOM.MFnDependencyNode(node).name()

        # create handle for attr plugs
        offset_plug = apiOM.MPlug(node, MatrixConstraintNode.input_offset_matrix_attr)
        parent_plug = apiOM.MPlug(node, MatrixConstraintNode.input_parent_inverse_matrix_attr)

        # create handles for attr input values
        parent_im_handle = data.inputValue(MatrixConstraintNode.input_parent_inverse_matrix_attr)
        input_use_offset_handle = data.inputValue(MatrixConstraintNode.input_use_offset_attr).asBool()
        input_offset_handle = data.inputValue(MatrixConstraintNode.input_offset_matrix_attr)
        input_compound_array_handle = data.inputArrayValue(MatrixConstraintNode.input_compound_attr)

        # create handles for attr output values
        output_translate = data.outputValue(MatrixConstraintNode.output_translate_attr)
        output_rotate = data.outputValue(MatrixConstraintNode.output_rotate_attr)

        # raise warning if parent inverse plug not connected
        if not parent_plug.isConnected:
            cmds.warning("'Parent Inverse Matrix' is not connected. May not evaluate as expected.")

        # if there is no previous offset, set it as og driven world matrix
        if self.offset is None:
            input_offset_handle.setMMatrix(self.og_wm)

        # use identity matrix for calculations if use offset False
        offset = apiOM.MMatrix()
        if input_use_offset_handle:
            offset = input_offset_handle.asMatrix()

        #
        drivers = []
        weights = []
        drivers_len = len(input_compound_array_handle)

        # for each driver get matrix and influence weight
        for index in range(drivers_len):

            input_compound_array_handle.jumpToPhysicalElement(index)

            input_compound_handle = input_compound_array_handle.inputValue()

            input_driver_handle = apiOM.MDataHandle(
                input_compound_handle.child(MatrixConstraintNode.input_driver_matrix_attr)).asMatrix()
            input_weight_handle = apiOM.MDataHandle(
                input_compound_handle.child(MatrixConstraintNode.input_driver_float_attr)).asFloat()

            drivers.append(input_driver_handle)
            weights.append(input_weight_handle)

        # get list of driver names
        new_driver_names = cmds.listConnections(node_name, destination=False, skipConversionNodes=True)
        if offset_plug.isConnected:
            new_driver_names = new_driver_names[:-1]

        # recompute individual driver offset if driver or offset changes
        if new_driver_names != self.driver_names or offset != self.offset:
            self.offset = offset
            self.driver_offsets = []
            for index, driver in enumerate(drivers):
                driver_offset = offset * driver.inverse()
                self.driver_offsets.append(driver_offset)

            self.driver_names = new_driver_names

        # compute driven matrix for each driver
        driven_matrices = []
        for index, driver in enumerate(drivers):
            if input_use_offset_handle:
                driven_matrix = self.driver_offsets[index] * driver * parent_im_handle.asMatrix()
            else:
                driven_matrix = driver * parent_im_handle.asMatrix()
            driven_matrices.append(driven_matrix)

        # if there are no drivers, set the translate and rotate output to the og driven ones
        if not driven_matrices:
            output_translate.set3Float(*self.og_pos)
            output_rotate.set3Float(*self.og_rot)
        # else blend drivers (calculate_output()) and set translate and rotate output
        else:
            pos, rot = calculate_output(driven_matrices, weights)

            output_translate.set3Float(*pos)
            output_rotate.set3Float(*rot)

        # set dirty plug clean
        data.setClean(plug)

    @staticmethod
    def on_delete(node, plug, self):

        """
        callback function on deletion
        disconnects output node plugs

        :param node: MObject - Node to query for callbacks
        :param plug: MPlug
        :param self: reference to self
        :return: None
        """

        node_name = apiOM.MFnDependencyNode(node).name()

        output_plugs = cmds.listConnections(node_name, plugs=True, connections=True, source=False, skipConversionNodes=True)
        if output_plugs:
            for index, plug in enumerate(output_plugs):
                if not index % 2:
                    cmds.disconnectAttr(plug, output_plugs[index + 1])


class MatrixConstraintCmd(apiOM.MPxCommand):

    """
    command for matrixConstraint node
    """

    cmd_name = "matrixConstraint"

    # flag names
    name_flag = "n", "name"
    weights_flag = "w", "weights"
    inverse_flag = "i", "inverse"
    use_offset_flag = "uo", "useOffset"
    pos_flag = "t", "translate"
    rot_flag = "r", "rotate"

    def __init__(self):
        super(MatrixConstraintCmd, self).__init__()

        self.node = None

        self.transforms = list()

        # set default values for args and flags
        self.objs = None
        self.name = MatrixConstraintCmd.cmd_name
        self.weights = list()
        self.inverse = False
        self.use_offset = True
        self.translate = "xyz"
        self.rotate = "xyz"

    def doIt(self, args):

        """
        query and store user input
        call redoIt() to build node

        :param args: MArgList
        :return: None
        """

        # try to get MArgDatabase from user input
        try:
            arg_db = apiOM.MArgDatabase(self.syntax(), args)
        # raise error if there is a problem with the input
        except RuntimeError:
            self.displayInfo("Error parsing arguments")
            raise

        # store args
        self.objs = arg_db.getObjectStrings()

        # override name flag if given
        if arg_db.isFlagSet(MatrixConstraintCmd.name_flag[0]):
            self.name = arg_db.flagArgumentString(MatrixConstraintCmd.name_flag[0], 0)

        # override inverse flag if given
        if arg_db.isFlagSet(MatrixConstraintCmd.inverse_flag[0]):
            self.inverse = arg_db.flagArgumentBool(MatrixConstraintCmd.inverse_flag[0], 0)

        # override use offset flag if given
        if arg_db.isFlagSet(MatrixConstraintCmd.use_offset_flag[0]):
            self.use_offset = arg_db.flagArgumentBool(MatrixConstraintCmd.use_offset_flag[0], 0)

        # fill weight flag with amount of drivers given
        driver_amount = len(self.objs) - 1
        if self.inverse:
            driver_amount = 1
        self.weights = [1 for _ in range(driver_amount)]

        # override weight flag if given
        if arg_db.isFlagSet(MatrixConstraintCmd.weights_flag[0]):

            self.weights = []
            weight_amount = arg_db.numberOfFlagUses(MatrixConstraintCmd.weights_flag[0])

            if weight_amount == driver_amount:
                for index in range(weight_amount):
                    weight = arg_db.getFlagArgumentList(MatrixConstraintCmd.weights_flag[0], index)
                    self.weights.append(weight.asFloat(0))
            else:
                cmds.error("'weights': does not have right amount of arguments ({} needed)".format(driver_amount))
                return

        # override translate flag if given
        if arg_db.isFlagSet(MatrixConstraintCmd.pos_flag[0]):

            self.translate = ""
            channels = arg_db.flagArgumentString(MatrixConstraintCmd.pos_flag[0], 0)

            for channel in channels:
                if channel in "xyz" and channel not in self.translate:
                    self.translate += channel
                else:
                    cmds.error("'translate': name '{}' is not defined".format(channels))

        # override rotate flag if given
        if arg_db.isFlagSet(MatrixConstraintCmd.rot_flag[0]):

            self.rotate = ""
            channels = arg_db.flagArgumentString(MatrixConstraintCmd.rot_flag[0], 0)

            for channel in channels:
                if channel in "xyz" and channel not in self.rotate:
                    self.rotate += channel
                else:
                    cmds.error("'rotate': name '{}' is not defined".format(channels))

        # store new translate and rotate flag values
        self.transforms = [self.translate, self.rotate]

        # call redoIt() to build node
        self.redoIt()

    def redoIt(self):

        """
        create node and connections

        :return: None
        """

        # if inverse is checked, all objs are being constrained to last obj
        if self.inverse:

            driver = self.objs[-1]
            driven = self.objs[:-1]

            for obj in driven:

                self.node = cmds.createNode("matrixConstraint", name=self.name)
                cmds.setAttr("{}.useOffset".format(self.node), self.use_offset)
                cmds.connectAttr("{}.worldMatrix".format(driver), "{}.matrixIn[0].matrix".format(self.node))
                cmds.connectAttr("{}.parentInverseMatrix".format(obj), "{}.parentInverseMatrix".format(self.node))

                for index, transform in enumerate(["translate", "rotate"]):
                    if len(self.transforms[index]) == 3:
                        cmds.connectAttr("{}.{}".format(self.node, transform), "{}.{}".format(obj, transform))
                    else:
                        for channel in self.transforms[index]:
                            cmds.connectAttr("{}.{}{}".format(self.node, transform, channel.upper()),
                                             "{}.{}{}".format(obj, transform, channel.upper()))

        # else all objs drive the last obj
        else:

            drivers = self.objs[:-1]
            driven = self.objs[-1]

            self.node = cmds.createNode("matrixConstraint", name=self.name)
            cmds.setAttr("{}.useOffset".format(self.node), self.use_offset)
            cmds.connectAttr("{}.parentInverseMatrix".format(driven), "{}.parentInverseMatrix".format(self.node))

            for index, driver in enumerate(drivers):
                cmds.connectAttr("{}.worldMatrix".format(driver), "{}.matrixIn[{}].matrix".format(self.node, index))
                cmds.setAttr("{}.matrixIn[{}].weight".format(self.node, index), self.weights[index])

            for index, transform in enumerate(["translate", "rotate"]):
                if len(self.transforms[index]) == 3:
                    cmds.connectAttr("{}.{}".format(self.node, transform), "{}.{}".format(driven, transform))
                else:
                    for channel in self.transforms[index]:
                        cmds.connectAttr("{}.{}{}".format(self.node, transform, channel.upper()),
                                         "{}.{}{}".format(driven, transform, channel.upper()))

        self.inverse = False
        self.use_offset = True


    def undoIt(self):
        """delete created node on undo"""
        cmds.delete(self.node)

    def isUndoable(self):
        """tell maya command supports undo"""
        return True

    @classmethod
    def creator(cls):
        return MatrixConstraintCmd()

    @classmethod
    def create_syntax(cls):

        """
        create args and flags for command

        :return: MSyntax
        """

        syntax = apiOM.MSyntax()

        syntax.setObjectType(apiOM.MSyntax.kStringObjects)
        syntax.setMinObjects(2)

        # create and add flags
        syntax.addFlag(MatrixConstraintCmd.name_flag[0], MatrixConstraintCmd.name_flag[1], apiOM.MSyntax.kString)
        syntax.addFlag(MatrixConstraintCmd.weights_flag[0], MatrixConstraintCmd.weights_flag[1], apiOM.MSyntax.kDouble)
        syntax.makeFlagMultiUse(MatrixConstraintCmd.weights_flag[1])

        syntax.addFlag(MatrixConstraintCmd.inverse_flag[0], MatrixConstraintCmd.inverse_flag[1], apiOM.MSyntax.kBoolean)
        syntax.addFlag(MatrixConstraintCmd.use_offset_flag[0], MatrixConstraintCmd.use_offset_flag[1], apiOM.MSyntax.kBoolean)

        syntax.addFlag(MatrixConstraintCmd.pos_flag[0], MatrixConstraintCmd.pos_flag[1], apiOM.MSyntax.kString)
        syntax.addFlag(MatrixConstraintCmd.rot_flag[0], MatrixConstraintCmd.rot_flag[1], apiOM.MSyntax.kString)

        return syntax


def initializePlugin(obj):

    """initializing plugin"""

    plugin = apiOM.MFnPlugin(obj, 'Antonia Beck', '1.0')

    plugin.registerNode(MatrixConstraintNode.plugin_name, MatrixConstraintNode.plugin_id, MatrixConstraintNode.creator,
                        MatrixConstraintNode.initialize)

    plugin.registerCommand(MatrixConstraintCmd.cmd_name, MatrixConstraintCmd.creator, MatrixConstraintCmd.create_syntax)


def uninitializePlugin(obj):

    """uninitializing plugin"""

    plugin = apiOM.MFnPlugin(obj)

    plugin.deregisterNode(MatrixConstraintNode.plugin_id)

    plugin.deregisterCommand(MatrixConstraintCmd.cmd_name)


def calculate_output(matrices, weights):

    """
    take matrices and blend their translate and rotate according to their influence weights

    :param matrices: List of MMatrices
    :param weights: List of floats
    :return: MVector, List of floats
    """

    # run normalization function on weights - returns list
    weights = normalize(weights)

    pos = []
    qua = []

    # get translate and rotate values from each matrix and store them in lists
    for m in matrices:
        tform_matrix = apiOM.MTransformationMatrix(m)
        pos.append(tform_matrix.translation(apiOM.MSpace.kWorld))
        qua.append(tform_matrix.rotation(asQuaternion=True))

    new_qua = apiOM.MQuaternion(0, 0, 0, 0)
    new_pos = apiOM.MVector()
    new_rot = []

    # blend all translate and rotate values
    for value_index, value in enumerate(weights):
        new_qua = new_qua.slerp(new_qua, qua[value_index], value)

        for channel_index, channel in enumerate(pos[value_index]):
            new_pos[channel_index] += value * channel

    # get degrees from radians
    for radian in list(new_qua.asEulerRotation()):
        new_rot.append(math.degrees(radian))

    return new_pos, new_rot


def normalize(values):

    """
    normalize a list of floats in the range 0 to 1

    :param values: List of floats
    :return: List of floats
    """

    # create a new list, where each element has the same value and which sum equals 1
    new_values = [1.0/len(values) for _ in range(len(values))]

    if sum(values) != 0:
        # reset new list
        new_values = []
        # get sum of the arg list
        value_sum = sum(values)

        for value in values:
            # append normalized arg values to new list
            new_values.append(float(value) / float(value_sum))

    return new_values
