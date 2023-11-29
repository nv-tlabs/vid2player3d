import os
import sys
import os.path as osp

sys.path.append(os.getcwd())
from lxml.etree import XMLParser, parse, SubElement
from lxml import etree
import numpy as np
import joblib


class Bone:
    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []  # bvh only
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []

        # asf specific
        self.dir = np.zeros(3)
        self.len = 0
        # bvh specific
        self.offset = np.zeros(3)

        # inferred info
        self.pos = np.zeros(3)
        self.end = np.zeros(3)


class Skeleton:
    def __init__(
        self, template_dir="/hdd/zen/dev/copycat/Copycat/assets/bigfoot_template_v1.pkl"
    ):
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.dof_name = ["x", "y", "z"]
        self.root = None
        self.template_geoms = None
        if osp.isfile(template_dir):
            self.template_geoms = joblib.load(template_dir)

    def forward_bvh(self, bone):
        if bone.parent:
            # bone.pos = bone.parent.pos + bone.offset
            bone.pos = bone.offset
        else:
            bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)

    def load_from_offsets(
        self,
        offsets,
        parents,
        scale,
        jrange,
        exclude_bones=None,
        channels=None,
        spec_channels=None,
    ):
        if channels is None:
            channels = ["x", "y", "z"]
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()

        joint_names = list(
            filter(lambda x: all([t not in x for t in exclude_bones]), offsets.keys())
        )
        dof_ind = {"x": 0, "y": 1, "z": 2}
        self.len_scale = scale
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = channels
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint
            
            bone.channels = (
                spec_channels[joint] if joint in spec_channels.keys() else channels
            )
            bone.dof_index = [dof_ind[x] for x in bone.channels]
            bone.offset = np.array(offsets[joint]) * self.len_scale
            bone.lb = np.rad2deg(jrange[joint][:, 0])
            bone.ub = np.rad2deg(jrange[joint][:, 1])


            self.bones.append(bone)
            self.name2bone[joint] = bone
        for bone in self.bones[1:]:
            parent_name = parents[bone.name]
            # print(parent_name)
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)
        # import pdb
        # pdb.set_trace()
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.end = bone.pos.copy() + 0.002
                for c_bone, p_bone in parents.items():
                    if p_bone == bone.name:
                        bone.end += np.array(offsets[c_bone]) * self.len_scale
                        break
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def write_xml(
        self,
        fname,
        template_fname="/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/template/humanoid_template_local.xml",
        offset=np.array([0, 0, 0]),
        ref_angles=None,
        bump_buffer=False,
    ):
        if ref_angles is None:
            ref_angles = {}
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")
        self.write_xml_bodynode(self.root, worldbody, offset, ref_angles)

        # create actuators
        actuators = tree.getroot().find("actuator")
        joints = worldbody.findall(".//joint")
        for joint in joints[1:]:
            name = joint.attrib["name"]
            attr = dict()
            attr["name"] = name
            attr["joint"] = name
            attr["gear"] = "1"
            SubElement(actuators, "motor", attr)
        if bump_buffer:
            SubElement(tree.getroot(), "size", {"njmax": "700", "nconmax": "200"})
        tree.write(fname, pretty_print=True)

    def write_str(
        self,
        template_fname="/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/template/humanoid_template_local.xml",
        offset=np.array([0, 0, 0]),
        ref_angles=None,
        bump_buffer=False,
    ):
        if ref_angles is None:
            ref_angles = {}
        parser = XMLParser(remove_blank_text=True)
        tree = parse(template_fname, parser=parser)
        worldbody = tree.getroot().find("worldbody")
        self.write_xml_bodynode(self.root, worldbody, offset, ref_angles)

        # create actuators
        actuators = tree.getroot().find("actuator")
        joints = worldbody.findall(".//joint")
        for joint in joints[1:]:
            name = joint.attrib["name"]
            attr = dict()
            attr["name"] = name
            attr["joint"] = name
            attr["gear"] = "500"
            SubElement(actuators, "motor", attr)
        if bump_buffer:
            SubElement(tree.getroot(), "size", {"njmax": "700", "nconmax": "200"})

        return etree.tostring(tree, pretty_print=False)

    def write_xml_bodynode(self, bone, parent_node, offset, ref_angles):
        attr = dict()
        attr["name"] = bone.name
        attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
        attr["user"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.end + offset))
        node = SubElement(parent_node, "body", attr)

        # write joints
        if bone.parent is None:
            j_attr = dict()
            j_attr["name"] = bone.name
            # j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
            j_attr["limited"] = "false"
            j_attr["type"] = "free"
            j_attr["armature"] = "0"
            j_attr["damping"] = "0"
            # j_attr["stiffness"] = "500"
            SubElement(node, "joint", j_attr)
        else:
            for i in range(len(bone.dof_index)):
                ind = bone.dof_index[i]
                axis = bone.orient[:, ind]
                j_attr = dict()
                j_attr["name"] = bone.name + "_" + self.dof_name[ind]
                j_attr["type"] = "hinge"
                j_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
                j_attr["axis"] = "{0:.4f} {1:.4f} {2:.4f}".format(*axis)
                j_attr["stiffness"] = "500"
                j_attr["damping"] = "50"
                j_attr["armature"] = "0.02"

                if i < len(bone.lb):
                    j_attr["range"] = "{0:.4f} {1:.4f}".format(bone.lb[i], bone.ub[i])
                else:
                    j_attr["range"] = "-180.0 180.0"
                if j_attr["name"] in ref_angles.keys():
                    j_attr["ref"] = f"{ref_angles[j_attr['name']]:.1f}"

                SubElement(node, "joint", j_attr)

        # write geometry
        if self.template_geoms is None or len(self.template_geoms[bone.name]) == 0:
            if bone.parent is None:
                g_attr = dict()
                g_attr["size"] = "0.0300"
                g_attr["type"] = "sphere"
                g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*(bone.pos + offset))
            else:
                e1 = np.zeros(3)
                e2 = bone.end.copy() + offset
                g_attr = dict()
                g_attr["size"] = "0.0100"
                if bone.name.endswith("3"):
                    g_attr["type"] = "sphere"
                    g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(
                        *(bone.pos + offset)
                    )
                else:
                    g_attr["type"] = "capsule"
                    g_attr[
                        "fromto"
                    ] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(
                        *np.concatenate([e1, e2])
                    )

                g_attr["contype"] = "1"
                g_attr["conaffinity"] = "1"

        else:
            g_attr = dict()
            template_attributes = self.template_geoms[bone.name][0]
            g_attr["type"] = template_attributes["type"]
            # g_attr["contype"] = template_attributes["contype"]
            # g_attr["conaffinity"] = template_attributes["conaffinity"]
            g_attr["contype"] = "1"
            g_attr["conaffinity"] = "1"
            g_attr["density"] = "500"
            e1 = np.zeros(3)
            e2 = bone.end.copy() + offset
            # template_attributes["start"]
            if g_attr["type"] == "capsule":
                g_attr[
                    "fromto"
                ] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.4f}".format(
                    *np.concatenate(
                        [e1, e2]
                    )
                )
                g_attr["size"] = "{0:.4f}".format(*template_attributes["size"])
            elif g_attr["type"] == "box":
                # g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(
                #     *template_attributes["start"]
                # )
                multiplier = np.linalg.norm(e2 - e1) / 0.0945
                pos = (e1 + e2) / 2
                if bone.name == "L_Toe" or bone.name == "R_Toe":
                    pos[1] += 0.05
                

                g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(*pos)

                g_attr["size"] = "{0:.4f} {1:.4f} {2:.4f}".format(
                    *template_attributes["size"] * multiplier
                )
                g_attr["quat"] = "{0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(
                    *template_attributes["rot"]
                )
            elif g_attr["type"] == "sphere":
                g_attr["size"] = "{0:.4f}".format(*template_attributes["size"])
                g_attr["pos"] = "{0:.4f} {1:.4f} {2:.4f}".format(
                    *np.zeros(3)
                )
        SubElement(node, "geom", g_attr)


        # write child bones
        for bone_c in bone.child:
            self.write_xml_bodynode(bone_c, node, offset, ref_angles)
