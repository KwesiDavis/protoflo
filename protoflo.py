# ProtoFlo - Flow-Based Programming experiments in Python
# Copyright (c) 2014 Jon Nordby <jononor@gmail.com>
# ProtoFlo may be freely distributed under the MIT license
##

import sys, os
import functools
import json
import subprocess
import httplib
import uuid
import urllib
import time
import rpyc
import socket

import numpy
from scipy import misc, ndimage
import matplotlib.pyplot as plt

class Port(object):
    def __init__(self):
        self.target = None
        self.value = None
        # Before a component starts working on its inputs, it checks to see if all 
        # in-ports have received some data. It is possible, however, that some 
        # in-ports are not connected and will never receive data. The following 
        # attribute knows which ports are connected and are expecting data.
        self.is_expecting = False

    def connect(self, process, portname):
        '''
        Tell a the given process to expect data, on a particular port name.
        
        Parameters:
            process - The target process object this port is attached to.
            portname - The name of a port on the target process.
        '''
        target = (process, portname)
        process.ports[portname].is_expecting = True
        self.target = target

class AsyncPort(Port):
    '''
    A Port that supports non-blocking send()'s when used as an out-port.
    '''
    def __init__(self):
        '''
        When data is sent from a port, the send() method is called; which in turn
        calls receive() on the port's target. The receive() call can be time 
        consuming. This port supports sending data in a way that is non-blocking
        and does not have to wait for the port's target to complete its recieve().
        '''
        super(AsyncPort, self).__init__()
        self.async_result = None

class Component(object):
    def __init__(self):
        self.ports = {}
        # The name of the process this component represents.
        self.name  = ''

    def receive(self, data, port):
        raise NotImplementedError

    def send(self, data, port):
        # Record the output value so SubNets can read results of their 
        # internal processes.
        self.ports[port].value = data
                
        target = self.ports[port].target
        if target:
            tgtnode, tgtport = target
            tgtnode.receive(data, tgtport)
        else:
            pass

class Unary(Component):
    @staticmethod
    def factory(func):
        return functools.partial(Unary, func)

    def __init__(self, function):
        Component.__init__(self)
        self._func = function
        self.ports = {
            "in": Port(),
            "out": Port()
        }

    def receive(self, data, port):
        self.send(self._func(data), "out")

class Nary(Component):
    @staticmethod
    def factory(inports, func):
        return functools.partial(Nary, inports, func)

    def __init__(self, inports, function):
        Component.__init__(self)
        self._func = function
        self.ports = {
            "out": Port()
        }
        self._inports = inports
        for name in inports:
            self.ports[name] = Port()

    def receive(self, data, port):
        # Store new data for @port
        p = self.ports.get(port, None)
        if not p:
            raise ValueError, 'No port named %s in Nary(X)' % port
        p.value = data

        # Re-evaluate function
        # TODO: allow None?
        args = [self.ports[n].value for n in self._inports]
        if not any(x is None for x in args):
            res = self._func(*args)
            self.send(res, "out")

class AsyncComponent(Component):
    '''
    A component with any number of named in-ports and out-ports.  This 
    component can perform asynchronous send() operations on its connected 
    out-ports.
    '''
    @staticmethod
    def factory(inports, outports, func):
        return functools.partial(AsyncComponent, inports, outports, func)
    
    def __init__(self, inports, outports, function):
        """
        Initialize component with in-ports and out-ports of type AsyncPort.
        
        Parameters:
            inports - A string list of in-port names.
            outports - A string list of out-port names.
            function - A callback to execute when data arrives on all connected
                       in-ports .
        """
        super(AsyncComponent, self).__init__()
        self._func = function        
        self._inports = inports
        for name in inports + outports:
            self.ports[name] = AsyncPort()
                
    @staticmethod
    def async_receive(process_poxy, *args, **kwargs):
        '''
        A non-blocking (or asynchronous) wrapper around the given process' 
        receive() method.
        
        Parameters:
            process_poxy - An RPyC proxy object representing a process in 
                           the graph.
            args - The positional arguments for receive()
            kwargs - The key word arguments for receive()
        
        Returns:
            An asynchronous result object that can access the return value of 
            receive() as soon as it becomes available.
        '''
        asyncReceiveFxn = rpyc.async(process_poxy.receive)
        asyncResult     = asyncReceiveFxn(*args, **kwargs)
        return asyncResult    
                
    def receive(self, data, port):
        '''
        Accept the given data and apply it to a port with the given port name.
        
        Parameters:
            data - An incoming information packet.
            port - The string name of an in-port on this component.
        '''
        # Store new data for @port
        p = self.ports.get(port, None)
        if not p:
            raise ValueError, 'No port named %s in AsyncComponent(X)' % port
        p.value = data
        
        # Re-evaluate function
        # TODO: allow None?
        # Note: In-ports may have default values or are simply optional. So, 
        #       only wait on connected in-ports that are expecting a value.
        connected_inports = dict([(str(n), self.ports[n].value) for n in self._inports if self.ports[n].is_expecting])
        if not any(x is None for x in connected_inports.values()):
            # We ignore the positional order of the args and leverage kwargs to
            # index in-ports by name instead.  We do the same with out-ports by
            # always returning a dict as the result of a process.
            results = self._func(**connected_inports)
            for outportItem in results.items():
                port, data = outportItem
                self.send(data, port)
 
    @staticmethod
    def is_proxy(obj):
        '''
        Determine whether the given object is "real" or a proxy object.
        
        Parameters:
            obj - A Python object.
            
        Returns:
            'True' if the given object is an RPyC proxy object.
        '''
        return issubclass(type(obj).__class__, rpyc.core.netref.NetrefMetaclass)
    
    @staticmethod
    def async_send(outports, data, port):
        '''
        For a given bundle of connected out-ports, the method sends the given
        data through the out-port with the given port name. 
        Note: For components with multiple connected out-ports, this method
              call is asynchronous on every pending out-port name except the 
              last one; at which point it blocks until all send() operations 
              have completed.
        
        Parameters:
            outports - A dictionary of connected Port's (index by port name). 
            data - An incoming information packet.
            port - The string name of an out-port on this component.     
        '''
        # Record output value so SubNets can read results of internal processes
        outports[port].value = data
        
        target = outports[port].target
        if target:
            tgtnode, tgtport = target
            # If the process that will receive this data can do its receive()
            # asynchronously, then use the more efficient async_send() to send data.
            if AsyncComponent.is_proxy(tgtnode):
                # Non-blocking call to async_receive()
                outports[port].async_result = AsyncComponent.async_receive(tgtnode, data, tgtport)
                # Only AFTER send() has been called on ALL *connected* outports, do
                # we wait for the entire group of asynchronous send()'s to complete.
                # Then we reset the component's asynchronous bookkeeping.
                if not [outport for outport in outports.values() if outport.async_result is None]:
                    for outport in outports.values():
                        outport.async_result.wait() # wait on dependent components
                        outport.async_result = None # reset
            else:
                # Blocking call to receive()
                tgtnode.receive(data, tgtport)
        else:
            pass
        
    def send(self, data, port):
        '''
        Sends the given data through the out-port with the given port name. 
        Note: For components with multiple connected out-ports, this method
              call is asynchronous on every pending out-port name except the 
              last one; at which point it blocks until all send() operations 
              have completed.
        
        Parameters:
            data - An incoming information packet.
            port - The string name of an out-port on this component.     
        '''
        outports = dict([(portname, portobj) for portname, portobj in self.ports.items() if not portname in self._inports])
        AsyncComponent.async_send(outports, data, port)
        
class SubNet(AsyncComponent):
    '''
    A bundle of Components wired together to behave as a single composite-component.   
    ''' 
    @staticmethod
    def factory(graphPath, parallel=False):
        return functools.partial(SubNet, graphPath, parallel)

    def __init__(self, graphPath, parallel=True):
        '''
        Given a component graph build a sub-network to handle the business
        logic of this component.  Optionally, the resultant sub-network can
        execute the contained processes serially or in-parallel. 
        
        Parameters:
            graphPath - The string file path to a graph representation of this 
                        sub-network. 
            parallel - When 'True', the default, this component creates a 
                       network in which every child component runs in a 
                       separate thread.
        '''
        self._graph = load_file(graphPath)
        # Get the graph's interface
        inports  = [] if not 'inports' in self._graph else self._graph['inports'].keys() 
        outports = [] if not 'outports' in self._graph else self._graph['outports'].keys()
        # Set the Component's interface
        super(SubNet, self).__init__(inports, outports, self.run)
        self.parallel = parallel

    def run(self, iips=None, **kwargs ):
        '''
        Given a some information packet data, send these data to the contained
        sub-network as a series of initial information packets (or IIPs). Allow
        these data to ripple through the sub-network, collect the results and 
        return the results in a bundle.
                
        Parameters:
            iips - A list of IIP tuples, of the form (data, process_name, 
                  port_name), to stimulate the sub-network. These IIPs are 
                  concatenated with any IIP's that were loaded from the 
                  original graph representation. The list can be empty. If it 
                  is 'None', the default, IIPs are derived from kwargs.
            kwargs - A dictionary of information packet data (indexed by 
                     exported in-port name); which are converted to IIPs and
                     sent to the sub-network. When this dict is supplied,
                     IIP's that were loaded from the original graph 
                     representation are ignored.
        
        Returns:
            A dictionary of the sub-network's results (indexed by exported
            out-port name) 
        '''
        graph = self._graph

        # If IIPs are not explicitly provided, assume we can generate IIPs from
        # the kwarg dict; which will contain data that should be sent to the given
        # pre-defined external in-port name.  
        if iips == None:
            # -- Remove existing IIPs
            graph['connections'] = filter(lambda x: 'src' in x,  graph['connections'])
            # -- Represent data inputs as auxiliary IIPs to the subnet 
            def input2iip( inputPortName ):
                process = graph['inports'][inputPortName]['process']
                port    = graph['inports'][inputPortName]['port']
                data    = kwargs[key]
                return (data, process, port)
            iips = [input2iip(key) for key in kwargs.keys()]

        # Execute SubNet and generate output
        with ParallelNetwork(self) as net:
            net.start(iips=iips)
            net.run_iteration()            
            # Write outputs by reading the results off the ports on the output 
            # interface and returning these data as the result of the subnet.
            retval = {}
            externalOutPortNames = set(self.ports.keys()) - set(self._inports)
            for externalOutPortName in externalOutPortNames:
                process = graph['outports'][externalOutPortName]['process']
                port    = graph['outports'][externalOutPortName]['port']
                data    = net._nodes[process].ports[port].value
                retval[externalOutPortName] = data
        return retval

components = {
    "Invert": Unary.factory(lambda obj: not obj),
    "IncrementOne": Unary.factory(lambda obj: obj+1),
    "WriteStdOut": Unary.factory(lambda obj: sys.stdout.write(obj)),
    "Str": Unary.factory(lambda obj: str(obj)),
    "Sleep": Unary.factory(lambda obj: time.sleep(obj)),

    "Add": Nary.factory(["a", "b"], lambda a,b: a+b),
    "Subtract": Nary.factory(["a", "b"], lambda a,b: a-b),
    "Multiply": Nary.factory(["a", "b"], lambda a,b: a*b),
    "Divide": Nary.factory(["a", "b"], lambda a,b: a/b),
    "Numpy/Array": Nary.factory(["values"], lambda values: numpy.asarray(values)),
    "Scipy/Lena": Nary.factory(["kick"], lambda ignore: misc.lena()),
    "Scipy/GaussianFilter": Nary.factory(["array", "sigma"], lambda a, s: ndimage.gaussian_filter(a, sigma=s)),
    "Plot/ImageShow": Nary.factory(["array", "colormap"], lambda a,c: plt.imshow(a, cmap=c)),
    "Plot/Show": Nary.factory(["kick"], lambda ignore: plt.show()),
    "Time": Nary.factory(["kick"], lambda ignore: time.time()),
    
    "Clone": AsyncComponent.factory(["in"], ["out", "out2"], lambda **kwargs: {"out":kwargs["in"], "out2":kwargs["in"]}),
    
    "SleepSubNet": SubNet.factory('./examples/sleep_subnet.json')
}

def map_literal(data):
    converters = [
        lambda d: int(data),
        lambda d: float(data),
        lambda d: d,
    ]
    for conv in converters:
        try:
            return conv(data)
        except (ValueError, TypeError), e:
            continue

    raise Error, 'Should never be reached'

class Network(object):
    def __init__(self, graph):
        self._graph = graph
        self.stop()

    def stop(self):
        self._state = "stopped"
        self._nodes = {}
        self._msgqueue = []

    def start(self, iips=[]):
        '''
        Create and then connect the processes in the graph.
        
        Parameters:
            iips - additional IIPs to stimulate the graph
        '''
        # Instantiate components
        graph = self._graph
        for name, data in graph['processes'].items():
            self._nodes[name] = self.get_process( data['component'] )
            # Tell every process its name
            self._nodes[name].name = name

        # Add some auxiliary IIPs
        more_iips = [ {'data':data, 'tgt':{'process':process, 'port':port}} for data, process, port in iips ]

        # Wire up ports, IIPs
        for conn in graph['connections'] + more_iips:
            tgt = conn['tgt']
            src = conn.get('src', None)
            data = conn.get('data', None)
            data = map_literal(data)
            if src:
                self.connect(src['process'], src['port'],
                             tgt['process'], tgt['port'])
            elif data is not None:
                iip = (tgt['process'], tgt['port'], data)
                self.send(*iip)
            else:
                raise ValueError, "No src node or IIP"
       
    def get_process(self, component_name):
        '''
        Instantiate a component of the given type and return it as a new process.
        
        Parameters:
            component_name - The class name of a component to create.
            
        Returns:
            A new process.
        '''
        return components[component_name]()

    def connect(self, src, srcport, tgt, tgtport):
        if not isinstance(src, Component):
            src = self._nodes[src]
        if not isinstance(tgt, Component):
            tgt = self._nodes[tgt]

        src.ports[srcport].connect(tgt, tgtport)
     
    def send(self, tgt, port, data):
        if not isinstance(tgt, Component):
            tgt = self._nodes[tgt]

        ip = (tgt, port, data)
        self._msgqueue.append(ip)

    def _deliver_messages(self):
        stop_index = len(self._msgqueue)
        for index, msg in enumerate(self._msgqueue[0:stop_index]):
            if index == stop_index:
                break
            tgt, port, data = msg
            tgt.receive(data, port)
        self._msgqueue = self._msgqueue[stop_index:]

    def run_iteration(self):
        self._deliver_messages()

        
class ParallelNetwork(Network):
    '''
    A network of processes operating in parallel.
    '''
    
    # The port on which the RPyC component service runs
    def __init__(self, subnet, url=('localhost', 8000)):
        '''
        Build a network of processes to simulate the behavior of the given 
        composite-component.
        
        Parameters:
            subnet - The process responsible for managing this network of 
                     sub-processes.
            url - The URL of the process-server that manages the parallel
                  processes on the network. It is a tuple of the form 
                  (hostname, port)
        '''
        super(ParallelNetwork, self).__init__(subnet._graph)
        self.subnet = subnet
        self.url = url

    def __enter__(self):
        '''
        Let a 'with' statement start the network's component server.
        '''
        self.start_server()
        return self
    
    def __exit__(self, type, value, traceback):
        '''
        When a 'with' block ends, do automatic resource cleanup.
        '''
        self.stop_server()

    def stop(self):
        '''
        Reset the network.
        '''
        super(ParallelNetwork, self).stop()
        self.server_process = None
        self.connections = []
        self.bg_serving_threads = []

    def _deliver_messages(self):
        '''
        Send all pending IIPs from the queue into the network.
        '''
        stop_index = len(self._msgqueue)

        # Queue all IIPs in this dict so async_send() knows how many 
        # IPs to send asynchronously before blocks and waits for results.
        newPort = lambda x : AsyncPort() if x else Port()
        iips = dict([(index,newPort(self.subnet.parallel)) for index in range(stop_index)])

        for index, msg in enumerate(self._msgqueue[0:stop_index]):
            if index == stop_index:
                break
            tgt, port, data = msg
            iips[index].connect(tgt, port)
            AsyncComponent.async_send(iips, data, index)
        self._msgqueue = self._msgqueue[stop_index:]

    def start_server(self):
        '''
        Start a service to manage a network of processes running in parallel.  
        '''
        host, port = self.url
        if self.subnet.parallel:
            try:
                # If we can connect to the a server, we don't need one.
                rpyc.classic.connect(host, port).close()
                return
            except socket.error:
                # RPyC throws annoying warnings in Python 2.6 
                import warnings
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                 
                def target(host, port, threads=True):
                    '''
                    Start a process that will generate a new thread for every 
                    connection. Each Component, in the graph, will be represented by a
                    new dedicated connection for the proxy object representing that 
                    component.
                    '''
                    from rpyc import SlaveService
                    if threads:
                        from rpyc.utils.server import ThreadedServer
                        t = ThreadedServer(SlaveService, hostname=host, port=port)
                    else:
                        from rpyc.utils.server import ForkingServer
                        t = ForkingServer(SlaveService, hostname=host, port=port)
                    t.start()
                # Spawn a process to deploy the server
                import multiprocessing
                args=(host, port, True)
                self.server_process = multiprocessing.Process(target=target, args=args)
                self.server_process.start()

    def stop_server(self):
        '''
        Release all connections and kill the component server.
        '''
        if self.subnet.parallel:
            for bgsrv in self.bg_serving_threads:
                bgsrv.stop()
            for conn in self.connections:
                conn.close()
            if self.server_process:
                self.server_process.terminate()

    def timeout(self, timeout, fxn, *args, **kwargs):
        '''
        Try executing the given function repeatedly until it either succeeds or
        the given timeout expires.
        
        Parameters:
            timeout - A maximum amount of time (in seconds) to try executing
                      the given function.
            fxn - A function to try to execute.
            args - Positional arguments for the given function
            kwargs - Key word arguments for the given function
        '''
        retval = None
        tStart = time.time()
        while True:
            try:
                retval = fxn(*args, **kwargs)
                break
            except Exception, e:
                if time.time()-tStart < timeout:
                    time.sleep(0.1)
                else:
                    raise e
        return retval

    def get_process(self, component_name):
        '''
        Instantiate a component of the given type, in its own thread, and 
        return proxy object that represents the new process. 
        
        Parameters:
            component_name - The class name of a component to create.
            
        Returns:
            A proxy object for a new process.
        '''
        if self.subnet.parallel:
            # Serve this component from a new thread
            host, port = self.url
            conn = self.timeout(1, rpyc.classic.connect, host, port)
            self.connections.append(conn)

            # We have process proxy objects (running in dedicated threads) 
            # making asynchronous method calls on other proxy objects; which 
            # they *contain* (in their Ports). This kind of setup requires the
            # use of background serving threads to maintain the parallel behavior. 
            bgsrv = rpyc.BgServingThread(conn)
            self.bg_serving_threads.append(bgsrv)
            # Create proxy object
            process = conn.modules.protoflo.components[component_name]()
            # Pass down parallel option to child SubNets
            try:
                process.parallel = self.subnet.parallel
            except AttributeError:
                pass
        else:
            process = components[component_name]()
        return process

def load_file(path):
    ext = os.path.splitext(path)[1]
    if ext == ".fbp":
        # TODO: implement natively. Using pyPEG/grako?
        s = subprocess.check_output(["fbp", path])
        return json.loads(s)
    elif ext == ".json":
        f = open(path, "r")
        return json.loads(f.read())
    else:
        raise ValueError, "Invalid format for file %s" % path


from autobahn.twisted.websocket import WebSocketServerProtocol, WebSocketServerFactory
from autobahn.websocket.compress import PerMessageDeflateOffer, PerMessageDeflateOfferAccept
from twisted.python import log
from twisted.internet import reactor


class NoFloUiProtocol(WebSocketServerProtocol):


    def onConnect(self, request):
        return 'noflo'

    def onOpen(self):
        self.sendPing()
        pass

    def onClose(self, wasClean, code, reason):
        pass

    def onMessage(self, payload, isBinary):
        if isBinary:
            raise ValueError, "WebSocket message must be UTF-8"

        cmd = json.loads(payload)
        print cmd

        if cmd['protocol'] == 'component' and cmd['command'] == 'list':
            for name, comp in components.items():
                c = comp()
                # FIXME: separate outports from inports
                inports = [{ "id": p, "type": "all" } for p in c.ports.keys() if not p == "out"]
                payload = { "name": name,
                        "description": "",
                        "inPorts": inports,
                        "outPorts": [ {"id": "out", "type": "all" } ],
                }
                resp = {"protocol": "component",
                    "command": "component",
                    "payload": payload,
                }
                self.sendMessage(json.dumps(resp))


def runtime(port):
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:"+str(port), debug = True)
    factory.protocol = NoFloUiProtocol

    # Required for Chromium ~33 and newer
    def accept(offers):
        for offer in offers:
            if isinstance(offer, PerMessageDeflateOffer):
                return PerMessageDeflateOfferAccept(offer)
    factory.setProtocolOptions(perMessageCompressionAccept = accept)

    reactor.listenTCP(port, factory)
    reactor.run()

def register(user_id, label, ip, port):

    runtime_id = str(uuid.uuid4())

    conn = httplib.HTTPConnection("api.flowhub.io", 80)
    conn.connect()

    url = "/runtimes/"+runtime_id
    headers = {"Content-type": "application/json"}
    data = {
        'type': 'protoflo', 'protocol': 'websocket',
        'address': ip+":"+str(port), 'id': runtime_id,
        'label': label, 'port': port, 'user': user_id,
        'secret': "122223333",
    }

    conn.request("PUT", url, json.dumps(data), headers)
    response = conn.getresponse()
    if not response.status == 201:
        raise ValueError("Could not create runtime " + str(response.status) + str(response.read()))
    else:
        print "Runtime registered with ID", runtime_id

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog=sys.argv[0])
    subparsers = parser.add_subparsers(dest='command', help='')

    parser_register = subparsers.add_parser('register', help='Register runtime with Flowhub')
    parser_register.add_argument('--user', type=str, help='User UUID to register runtime for', required=True)
    parser_register.add_argument('--label', type=str, help='Label to use in UI for this runtime', default="ProtoFlo")
    parser_register.add_argument('--ip', type=str, help='WebSocket IP for runtime', default='ws://localhost')
    parser_register.add_argument('--port', type=int, help='WebSocket port for runtime', default=3569)

    parser_runtime = subparsers.add_parser('runtime', help='Start runtime')
    parser_runtime.add_argument('--port', type=int, help='WebSocket port for runtime', default=3569)

    parser_run = subparsers.add_parser('run', help='Run a graph non-interactively')
    parser_run.add_argument('--file', type=str, help='Graph file .fbp|.json', required=True)
    parser_run.add_argument('--parallel', help='Process components in parallel.', action="store_true")
    parser_run.add_argument('--iip', nargs=3, action='append', help='Send an IIP to the network: <data> <process> <port>', default=[])

    args = parser.parse_args(sys.argv[1:])
    if args.command == 'register':
        register(args.user, args.label, args.ip, args.port)
    elif args.command == 'runtime':
        runtime(args.port)
    elif args.command == 'run':
        # Load the given file as a SubNet and run it!
        mainProcess = SubNet.factory(args.file, parallel=args.parallel)()
        mainProcess.run(iips=args.iip)