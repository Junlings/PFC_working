import numpy as np
import matplotlib.pylab as plt
import StringIO
import math


def store_force(*args):
    """This function is called during |pfc| cycling to record the stress
    strain curve.
    """
    if it.cycle() % 100: return

    strain.append(vel * it.mech_age() * 100.0 / box_len)
    stress.append(abs(ba.force_unbal()[boundary_mask][:, 2]).sum() / 2.0 / area / 1e6)

class contactdict():
    def __init__(self):
        self.default = None
        self.cmatdict = {}

    def add_default(self,cmat):
        self.default = cmat

    def add(self,key,cmat):
        self.cmatdict[key] = cmat

    def cmd_cmat_default(self,wer):
        wer.write('cmat default ')
        if self.default.type != None:
            wer.write('type %s...\n' % (self.default.type))
        wer.write('model %s ' % (self.default.model))
        self.default.cmd_contact_model(wer)
        # if self.group != None:
            #self.wer.write('range group %s' % self.range)

    def cmd_cmat_add(self):
        self.wer.write('cmat add %i model %s.../n' % (self.counter,self.modelname))
        self.wer.write(self.get_cmat_property())
        if self.range != None:
            self.wer.write('range group %s' % self.range)

class contact(object):
    # base class
    def __init__(self,type=None,model=None):
        self.type = type    # ball-ball, etc..
        self.model = model  # linear, linearbond , etc





class cm_linearpbond(contact):

    def __init__(self):
        self.modelname = 'linearpbond'
        self.para = kargs



class cm_linear(contact):
    def __init__(self,kn=0,ks=0,fric=0,rgap = 0,lin_mode=0):
        super(cm_linear, self).__init__(type=None,model='linear')
        self.kn = kn
        self.ks = ks
        self.fric = fric
        self.rgap = rgap
        self.lin_mode = lin_mode

    def cmd_contact_model(self,wer):

        #wer.write('contact model kn %s {ks} ' % self)
        wer.write('property kn %s ks %s fric %s rgap %s lin_mode %s' % (self.kn,
                                                                             self.ks,
                                                                             self.fric,
                                                                             self.rgap,
                                                                             self.lin_mode))



class grading:
    def __init__(self,sievesizetable, cumulativepassingtable=None, retensiontable=None,minidia=0.001):
        self.SST = sievesizetable
        self.CPT = cumulativepassingtable
        self.RT = retensiontable
        self.wer = StringIO.StringIO()
        self.minidia  = minidia             # update the minimum sieve size


        if self.RT == None:
            self.RT = np.zeros(np.size(self.CPT))
            self.cal_RT()

    def cal_RT(self):
        for i in range(0,len(self.CPT)):
            if i == 0:
                self.RT[i] = self.CPT[i]
            else:
                self.RT[i] = self.CPT[i] - self.CPT[i-1]

    def cmd_export_box(self,wer,nb,rangemin,rangemax,porosity):
        wer.write('set random %i\n' % nb)
        wer.write('ball distribute box %f %f' % (rangemin,rangemax))
        wer.write('    porosity %f' % porosity)
        wer.write('    numbin %i ' % len(self.RT))

        for i in range(0,len(self.RT)):
            wer.write('    bin %i' % int(i+1))
            if i == 0:
                wer.write('        radius %f %f' % (self.minidia/2.0, self.SST[i]/2.0))
            else:
                wer.write('        radius %f %f' % (self.SST[i-1]/2.0, self.SST[i]/2.0))

            wer.write('        volumefraction %f' % self.RT[i])
        wer.write('\n')

    def export_to_file(self,filename):

        f1 = open(filename,'w+')
        f1.writelines(self.wer.getvalue())
        f1.close()

class PFC3D:
    def __init__(self):
        self.wer = StringIO.StringIO()
        self.contactlib = {}

    def initiation(self, xr, yr, zr):
        # setup the scale of the project
        self.wer.write('new\n')
        self.wer.write('domain extent %f %f %f %f %f %f\n' % (-xr, xr, -yr, yr, -zr, zr))





    def create_ball_from_file_wb(self,filename):
        # create ball from file import, data file shall be four column (x,y,z,iffix)
        data = np.loadtxt(filename, delimiter=",")
        for datum in data:
            r, p, fix = datum[0], datum[1:4], datum[4]
            b = balls.create(r, p)
            if fix == 1.0:
                if b.pos_z() > 0.0:
                    b.set_extra(1, 1)
                else:
                    b.set_extra(1, 2)
            else:
                b.set_extra(1, 0)


    def set_callback(self):
        it.set_callback("store_force", 43.0)
        it.command("cycle 40000")

    def run_cycle(self,n):
        cmd = "cycle %i" % n
        it.command(cmd)

class ball():
    def __init__(self,x,y,z,r,grouplabel=None,groupseq=None,id=None):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.id = id
        self.grouplabel = grouplabel
        self.groupseq = groupseq
        self.appliedforce = [0,0,0]
        self.appliedmoment = [0, 0, 0]
        self.contactforce = [0, 0, 0]
        self.contactmoment = [0, 0, 0]
        self.damp = 0
        self.density = 0
        self.displacement = [0,0,0]
        self.position = [self.x, self.y, self.z]
        self.velocity = [0,0,0]
        self.spin = [0,0,0]
        self.euler = [0,0,0]



class balldict():
    def __init__(self):
        self.balldict = {}
        self.groupdict = {}
        self.groupproperty = {}
        self.idproperty = {}
        self.wer = StringIO.StringIO()

    def add(self,key,ball):
        self.balldict[key] = ball

    def cmd_single_ball_create(self,wer,x,y,z,r,group=None,groupseq=None,id=None):

        wer.write('ball create x %f  y %f z %f radius %f ' %  (x, y, z, r))

        if group != None:
            wer.write('group %s slot %i ' % (group,groupseq))

        if id != None:
            wer.write('id %i\n' % int(id))

        wer.write('\n')

    def cmd_dict_ball_create(self,wer):
        for key,ball in self.balldict.items():
            self.cmd_single_ball_create(wer,ball.x,ball.y,ball.z,ball.r,id=key,group=ball.grouplabel,groupseq=ball.groupseq)

    def cmd_dict_ball_attribute(self,wer):
        for groupkey in self.groupproperty.keys():
            #print self.groupproperty[groupkey]
            wer.write('ball attribute ')

            for propkey,prop in self.groupproperty[groupkey].items():
                wer.write('%s %s ' % (propkey, str(prop)))

            wer.write('range group %s \n' % (groupkey))

        for idkey in self.idproperty.keys():
            #print self.idproperty[idkey ]
            wer.write('ball attribute ')

            for propkey,prop in self.idproperty[idkey].items():
                wer.write('%s %s ' % (propkey, str(prop)))

            wer.write('range id %s \n' % (idkey))

    def cmd_single_ball_fix(self,id,wer,DOFs):
        wer.write('ball fix ')
        for item in DOFs:
            wer.write('%s ' % item)
        wer.write('range id %s \n' % id)


    def setpropbygroup(self,grouplabel,propname,propvalue):
        if grouplabel in self.groupproperty.keys():
            self.groupproperty[grouplabel].update({propname:propvalue})
        else:
            self.groupproperty.update({grouplabel:{propname:propvalue}})

    def setpropbyid(self,id,propname,propvalue):
        if id in self.idproperty.keys():
            self.idproperty[id].update({propname:propvalue})
        else:
            self.idproperty.update({id:{propname:propvalue}})


    def generate_SC(self,dimX = 1, dimY = 1, dimZ = 1, radiusBall = 0.1, latticeConst = 1.0,group=None,orign=[0,0,0]):
        n = len(self.balldict.keys())+1
        for x in xrange(0, dimX + 1):
            for y in xrange(0, dimY + 1):
                for z in xrange(0, dimZ + 1):
                    self.add(n, ball(orign[0] + x * latticeConst,
                                          orign[1] + y * latticeConst,
                                          orign[2] + z * latticeConst, radiusBall))

                    if group != None:
                        if group in self.groupdict.keys():
                            self.groupdict[group].append(n)
                        else:
                            self.groupdict[group]=[n]

                        self.balldict[n].grouplabel = group
                        self.balldict[n].groupseq = len(self.groupdict[group])


                    n = n + 1


    def generate_BCC(self,dimX = 1, dimY = 1, dimZ = 1, radiusBall1 = 0.1,radiusBall2 = 0.1,latticeConst = 1.0):

        self.generate_SC(dimX, dimY, dimZ, radiusBall1,latticeConst,group='1')
        self.generate_SC(dimX-1, dimY-1, dimZ-1, radiusBall2,latticeConst,group='2',orign=[latticeConst/2.0,
                                                                                     latticeConst/2.0,
                                                                                     latticeConst/2.0])


'''
# Step 1 =====================   initial the domain, setup the material properties
it.command("""
new
domain extent -1 1 -1 1 -1 1
cmat default model linearpbond property pb_ten 12.0e6 pb_coh 10.0e6 pb_fa 20.0 fric 0.5  ...
                   method deformability    emod 8.5e9 kratio 2.5                         ...
                   method pb_deformability emod 8.5e9 kratio 2.5                         ...
                   proximity 4e-4
""")


# Step 2 =====================   generate the balls by import the files
box_len = 38.1e-3


# Step 3 ===================== setup contact properties
it.command("""
ball attribute density 2000.0 damp 0.6
clean
contact method bond gap 4e-4
contact property lin_mode 1
cmat default proximity 0.0
clean all
""")

print "This model has {} contacts".format(it.contact.count())



# Step 4 ===================== add top and bottom bounary
vel = 1e-2
top_count = 0
bottom_count = 0
for b in balls.list():
    if b.extra(1) == 1:
        b.set_vel_z(-vel)
        b.set_fix(3, True)
        top_count += 1
    elif b.extra(1) == 2:
        b.set_vel_z(0)
        b.set_fix(3,True)
        bottom_count += 1

print "{} top boundary particles and {} bottom boundary particles".format(top_count, bottom_count)


r"""Create some (empty) lists to store the time and unbalanced forces
during the UCS test. """

top_mask = np.array([b.extra(1) == 1 for b in balls.list()])
bottom_mask  = np.array([b.extra(1) == 2 for b in balls.list()])
boundary_mask = np.logical_or(top_mask, bottom_mask)

r"""Create some (empty) lists to store the time and unbalanced forces
during the UCS test. """

strain = []
stress = []
area = box_len**2

# Step 5 ===================== setup the callback function to collect information during the cycling.



# Step 6 ===================== Issue the callback function and trigger the cycling



# Step 7 ===================== PostProcess

# obtain the peak stress and
speak = np.amax(stress)
ipeak = np.argmax(stress)   # this is the index of the maximun stress
epeak = strain[ipeak]/100.0

i50 = int(0.5*ipeak)       # use half of the maximum stress for calculate the modulus of elasticity, secant modulus
s50 = stress[i50]*1e6
e50 = strain[i50]/100.0
emod50 = s50 / e50 / 1e9

txtE = r'$E_{{50}}={:.1f} GPa$'.format(emod50)
txtS = r'$UCS = {:.1f} MPa$'.format(speak)


r"""We use the matplotlib.pylab module to create a plot of the stress versus strain response of the model """
plt.plot(strain, stress)
plt.title("Stress vs Strain")
plt.xlabel("Strain [%]")
plt.ylabel("Stress [MPa]")
plt.grid(True)
plt.text(0.25*100*e50,0.5*speak, txtE,fontsize=12)
plt.text(0.75*100*epeak,1.01*speak, txtS,fontsize=12)
plt.draw()
plt.savefig('p3d-test-ucs.png')
plt.close('all')
'''

def create_sample():
    passing_table = np.array([[0.03, 0.083], [0.04, 0.166], [0.05, 0.26],
                              [0.06, 0.345], [0.075, 0.493], [0.1, 0.693],
                              [0.12, 0.888], [0.15, 0.976], [0.18, 0.997], [0.25, 1.0]])
    GT1 = grading(passing_table[:, 0], cumulativepassingtable=passing_table[:, 1], minidia=0.001)
    GT1.cmd_export_box(10001, -1, 1, 0.6)
    GT1.export_to_file('tt.dat')

if __name__ == "__main__":
    #b1 = balldict()
    #a = 0.1
    #r = math.sqrt(3) * a / 4
    #b1.generate_BCC(3, 3, 3, radiusBall1=r, radiusBall2=r, latticeConst=a)
    #b1.cmd_dict_ball_create()

    #b1.setpropbygroup('1','density',2000)
    #b1.setpropbygroup('1', 'damp', 0.6)
    #b1.setpropbygroup('2', 'density', 2000)
    #b1.setpropbygroup('2', 'damp', 0.6)
    #b1.cmd_dict_ball_attribute()

    work1 = PFC3D()
    work1.initiation(1, 1, 1)

    b1 = balldict()
    b1.add(1, ball(0, 0, 0.25, 0.25, grouplabel='bottom', groupseq=1,id=1))
    b1.add(2, ball(0, 0, 0.75, 0.25, grouplabel='top', groupseq=2,id=2))
    b1.cmd_dict_ball_create(work1.wer)

    b1.setpropbygroup('bottom', 'density', 2000)
    b1.setpropbygroup('bottom', 'damp', 0.6)
    b1.setpropbygroup('top', 'density', 2000)
    b1.setpropbygroup('top', 'damp', 0.6)

    b1.setpropbyid(2, 'zdisplacement', 0.01)

    b1.cmd_dict_ball_attribute(work1.wer)
    b1.cmd_single_ball_fix(1, work1.wer, ['velocity'])

    ctd = contactdict()
    ct1 = cm_linear(kn=90, ks=10, fric=0.4, rgap=0.01, lin_mode=0)
    ctd.add_default(ct1)
    ctd.cmd_cmat_default(work1.wer)

    print work1.wer.getvalue()

