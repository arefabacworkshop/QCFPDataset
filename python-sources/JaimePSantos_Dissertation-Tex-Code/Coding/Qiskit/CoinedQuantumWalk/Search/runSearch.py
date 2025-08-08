import sys
sys.path.append('../../Tools')
from IBMTools import( 
        simul,
        savefig,
        saveMultipleHist,
        printDict,
        plotMultipleQiskit,
        plotMultipleQiskitGrover)
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from qiskit import( ClassicalRegister,
        QuantumRegister,
        QuantumCircuit,
        execute,
        Aer,
        transpile
        )
from qiskit.visualization import( plot_histogram,
        plot_state_city)
plt.rcParams['figure.figsize'] = 11,8
matplotlib.rcParams.update({'font.size' : 15})

def bipartiteWalk(N,n,qreg,qcoin):
    qreg = QuantumRegister(N)
    qcoin = QuantumRegister(n)
    qc = QuantumCircuit(qreg,qcoin,name='BipartiteGraph')
    qc.x(qreg[N-1])
    qc.swap(qreg[0:N-1],qcoin[0:n])
    return qc

def completeGraphWalk(N):
    qreg = QuantumRegister(N)
    qcoin = QuantumRegister(N)
    qc = QuantumCircuit(qreg,qcoin,name='    Shift    ')
    qc.swap(qreg[0:N],qcoin)
    return qc

def completeGraphWalkHCoin(N):
    qreg = QuantumRegister(N,'vertices')
    qcoin = QuantumRegister(N,'coin')
    qc = QuantumCircuit(qreg,qcoin,name='CompleteGraph')
    qc.h(qcoin)
    qc.swap(qreg[0:N],qcoin)
    return qc

def markedListComplete(markedList,N):
    oracleList = np.ones(2**N)
    for element in markedList:
        oracleList[element] = -1
    oracleList = oracleList*np.exp(1j*2*np.pi)
    return oracleList.tolist()

def diffusionComplete(N):
    qreg = QuantumRegister(N)
    qcoin = QuantumRegister(N)
    difCirc = QuantumCircuit(qreg,qcoin,name='Diffusion')
    difCirc.h(qcoin)
    aux = markedListComplete([0],N)
    qcAux = oracleComplete(aux,N,True)
    difCirc.append(qcAux,range(2*N))
    difCirc.h(qcoin)
    difCirc = transpile(difCirc,basis_gates=['cx','u3'],optimization_level=3)
    return difCirc

def drawCircDiffusionComplete(N):
    qreg = QuantumRegister(N)
    qcoin = QuantumRegister(N)
    difCirc = QuantumCircuit(qreg,qcoin,name='     Diff     ')
    difCirc.h(qcoin)
    aux = markedListComplete([0],N)
    qcAux = oracleComplete(aux,N,True)
    difCirc.append(qcAux,range(2*N))
    difCirc.h(qcoin)
    difCirc = transpile(difCirc)#,basis_gates=['cx','u3'],optimization_level=3)
    return difCirc

def oracleComplete(markedList,N,dif):
    qreg = QuantumRegister(N)
    qcoin = QuantumRegister(N)
    qc = QuantumCircuit(qreg,qcoin,name='    Oracle     ')
    if(dif==True):
        qc.diagonal(markedList,qcoin)
    else:
        qc.diagonal(markedList,qreg)
    qc = transpile(qc,basis_gates=['cx','u3'],optimization_level=3)
    return qc

def drawCircOracleComplete(markedList,N,dif):
    qreg = QuantumRegister(N)
    qcoin = QuantumRegister(N)
    qc = QuantumCircuit(qreg,qcoin,name='    Oracle     ')
    if(dif==True):
        qc.diagonal(markedList,qcoin)
    else:
        qc.diagonal(markedList,qreg)
    qc = transpile(qc)#,basis_gates=['cx','u3'],optimization_level=3)
    return qc

def runSearchComplete(N,steps,markedVertex):
    qreg = QuantumRegister(N,'vertices')
    qcoin = QuantumRegister(N,'coin')
    creg = ClassicalRegister(N)
    qc = QuantumCircuit(qreg,qcoin,creg)
    markedVertex=markedListComplete(markedVertex,N)
    qcOracle = oracleComplete(markedVertex,N,False)
    qcDif = diffusionComplete(N)
    qcQWalk = completeGraphWalk(N)
    qc.h(qreg)
    for i in range(steps):
        qc.append(qcOracle,range(2*N))
        qc.append(qcDif,range(2*N))
        qc.append(qcQWalk,range(2*N))
    qc = transpile(qc,basis_gates=['cx','u3'],optimization_level=1)
    qc.measure(range(N),range(N))
    return qc

def saveCoinedSearchFig(N,steps,markedVertex,fig, filePath, defaultFileName):
    specificFileName = ""
    i=0
    for n,m in zip(N,markedVertex):
        specificFileName+= "N%s_M%s_S"%(n,m)
        for step in steps:
            specificFileName+="%s"%step
        i+=1
        if(len(N)-i==0):
            break
        specificFileName+="_"
    savefig(fig, filePath,defaultFileName+specificFileName)
    return specificFileName

def runMultipleSearchComplete(N,steps,markedVertex):
    "Creates several instances of the coined quantum walk search circuit."
    circList = []
    circListAux = []
    for n in N:
        qreg = QuantumRegister(n)
        qsub = QuantumRegister(1)
        creg = ClassicalRegister(n)
        for step in steps:
            circ = QuantumCircuit(qreg,qsub,creg)
            circ = runSearchComplete(n,step,markedVertex)
            circListAux.append(circ)
        circList.append(circListAux)
        circListAux = []
    return circList

def drawSearchComplete(N,steps,markedVertex,style):
    qreg = QuantumRegister(N,'qv')
    qcoin = QuantumRegister(N,'qc')
    creg = ClassicalRegister(N)
    qc = QuantumCircuit(qreg,qcoin,creg)
    markedVertex=markedListComplete(markedVertex,N)
    qcOracle = drawCircOracleComplete(markedVertex,N,False)
    qcDif = drawCircDiffusionComplete(N)
    qcQWalk = completeGraphWalk(N)
    qc.h(qreg)
    #qc.barrier()
    for i in range(steps):
        qc.append(qcOracle,range(2*N))
        #qc.combine(qcOracle)
        qc.append(qcDif,range(2*N))
        qc.append(qcQWalk,range(2*N))
        qc.barrier()
    qc.measure(range(N),range(N))
    qc = transpile(qc)
    fig = qc.draw(output='mpl',style=style,fold=-1)
    return fig

def drawOracle(markedList,N,dif,style):
    qreg = QuantumRegister(N,'qv')
    qcoin = QuantumRegister(N,'qc')
    qc = QuantumCircuit(qreg,qcoin,name='    Oracle     ')
    if(dif==True):
        qc.diagonal(markedListComplete(markedList,N),qcoin)
    else:
        qc.diagonal(markedListComplete(markedList,N),qreg)
    qc = transpile(qc,basis_gates=['cx','rz','ccx','x','h'])
    fig = qc.draw(output='mpl',style=style,fold=-1)
    return qc

def drawDiffusion(N,style):
    qreg = QuantumRegister(N)
    qcoin = QuantumRegister(N)
    difCirc = QuantumCircuit(qreg,qcoin,name='     Diff     ')
    difCirc.h(qcoin)
    aux = markedListComplete([0],N)
    qcAux = drawCircOracleComplete(aux,N,True)
    difCirc.append(qcAux,range(2*N))
    difCirc.h(qcoin)
    difCirc = transpile(difCirc,basis_gates=['cx','rz','ccx','x','h'])#,basis_gates=['cx','u3'],optimization_level=3)
    fig = difCirc.draw(output='mpl',style=style,fold=-1)
    return fig

def saveCoinedSearchFig(N,steps,markedVertex,fig, filePath, defaultFileName):
    specificFileName = ""
    i=0
    for n,m in zip(N,markedVertex):
        specificFileName+= "N%s_M%s_S"%(n,m)
        for step in steps:
            specificFileName+="%s"%step
        i+=1
        if(len(N)-i==0):
            break
        specificFileName+="_"
    savefig(fig, filePath,defaultFileName+specificFileName)
    plt.clf()
    return specificFileName

def drawFlipFlopShift(N,style):
    qreg = QuantumRegister(N,'qv')
    qcoin = QuantumRegister(N,'qc')
    qc = QuantumCircuit(qreg,qcoin,name='    Shift    ')
    qc.swap(qreg[0:N],qcoin)
    qc = transpile(qc,basis_gates=['rz','x','h','swap'])
    fig = qc.draw(output='mpl',style=style)
    return fig 

filePath = 'CoinedQuantumWalk/Search/'
defaultFileName = "CoinedQiskitSearch_"
circFilePath = 'CoinedQuantumWalk/Search/Circuits/'
defaultCircFileName = "CoinedSearchQiskitCirc_"
defaultCircOracleFileName = "CoinedSearchQiskitCircOracle_"
defaultCircDiffFileName = "CoinedSearchQiskitCircDiff_"
defaultCircShiftFileName = "CoinedSearchQiskitCircShift_"

style = {'figwidth':38,'fontsize':16,'subfontsize':14}#,'compress':True}
styleOracle = {'figwidth':13,'fontsize':17,'subfontsize':14}#,'compress':True}
styleShift = {'figwidth':5,'fontsize':17,'subfontsize':14}#,'compress':True}
 
singleN = 3
singleSteps = 5
#fig = drawSearchComplete(singleN,singleSteps,[4],style)
#saveCoinedSearchFig([singleN],[singleSteps],[4],fig,circFilePath,defaultCircFileName)
#plt.show()
#fig2 = drawOracle([4],singleN,False,styleOracle)
#saveCoinedSearchFig([singleN],[singleSteps],[4],fig2,circFilePath,defaultCircOracleFileName)
#fig3 = drawDiffusion(singleN,styleOracle)
#saveCoinedSearchFig([singleN],[singleSteps],[4],fig3,circFilePath,defaultCircDiffFileName)
fig4= drawFlipFlopShift(singleN,styleShift)
saveCoinedSearchFig([singleN],[singleSteps],[4],fig4,circFilePath,defaultCircShiftFileName)
##TODO: markedVertex labels are not correct due to post processing.
#N=[3]
#steps=[0,2,4,5]
#markedVertex = [4] 
#shots = 3000
#multipleWalks = runMultipleSearchComplete(N,steps,markedVertex)
##fig = plotMultipleQiskit(N,multipleWalks,steps,shots,True)
##saveCoinedSearchFig(N,steps,markedVertex,fig,filePath,defaultFileName)
#fig5 = plotMultipleQiskitGrover(N,multipleWalks,steps,shots,True)
#saveCoinedSearchFig(N,steps,markedVertex,fig5,filePath,defaultFileName)
