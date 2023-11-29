import datetime
from lxml.etree import XMLParser, parse, Element
from copy import deepcopy


class AverageMeter(object):

    def __init__(self, avg=None, count=1):
        self.reset()
        if avg is not None:
            self.val = avg
            self.avg = avg
            self.count = count
            self.sum = avg * count
    
    def __repr__(self) -> str:
        return f'{self.avg: .4f}'

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def get_eta_str(cur_iter, total_iter, time_per_iter):
    eta = time_per_iter * (total_iter - cur_iter - 1)
    return convert_sec_to_time(eta)


def convert_sec_to_time(secs):
    return str(datetime.timedelta(seconds=round(secs)))


def create_vis_model_xml(in_file, out_file, num_actor=2, num_vis_capsules=0, num_vis_spheres=0, num_vis_planes=0):
    parser = XMLParser(remove_blank_text=True)
    tree = parse(in_file, parser=parser)
    geom_capsule = Element('geom', attrib={'fromto': '0 0 -10000 0 0 -9999', 'size': '0.02', 'type': 'capsule'})
    geom_sphere = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.02', 'type': 'sphere'})
    geom_plane = Element('geom', attrib={'pos': '0 0 -10000', 'size': '0.15 0.15 0.005', 'type': 'box'})

    root = tree.getroot().find('worldbody')
    body = root.find('body')
    for body_node in body.findall('.//body'):
        for joint_node in body_node.findall('joint')[1:]:
            body_node.remove(joint_node)
            body_node.insert(0, joint_node)

    for i in range(1, num_actor):
        new_body = deepcopy(body)
        new_body.attrib['childclass'] = f'actor{i}'
        new_body.attrib['name'] = '%d_%s' % (i, new_body.attrib['name'])
        for node in new_body.findall(".//body"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//joint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        for node in new_body.findall(".//freejoint"):
            node.attrib['name'] = '%d_%s' % (i, node.attrib['name'])
        root.append(new_body)
    act_node = tree.find('actuator')
    act_node.getparent().remove(act_node)

    ind = 2
    for i in range(num_vis_capsules):
        root.insert(ind, deepcopy(geom_capsule))
        ind += 1
    for i in range(num_vis_spheres):
        root.insert(ind, deepcopy(geom_sphere))
        ind += 1
    for i in range(num_vis_planes):
        root.insert(ind, deepcopy(geom_plane))
        ind += 1
    
    tree.write(out_file, pretty_print=True)