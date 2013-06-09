#
# Copyright (c) 2009 CodeSourcery
# Copyright (c) 2013 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.GPL file.

import os
import qm
from   qm.fields import *
from   qm.executable import *
from   qm.extension import parse_descriptor
import qm.test.base
from   qm.test.base import get_extension_class
from   qm.test.test import *
from   qm.test.resource import Resource
from   qm.test.database import get_database
from   qm.test.database import NoSuchTestError
from   qm.test.parameter_database import ParameterDatabase
from   qm.test.classes.explicit_suite import ExplicitSuite
from   qm.test.classes.compilation_test import ExecutableTest
from   qm.test.classes.compilation_test_database import CompilationTestDatabase
from   qm.test.classes.compilation_test_database import CompilationTest as CTBase
from   qm.test.classes.compiler_test import CompilerTest 
from   qm.test.classes.compiler_table import CompilerTable 
from   qm.test.classes.python import ExecTest as PythonExecTest
from   qm.test.directory_suite import DirectorySuite
from   qm.test.classes.command_host import CommandHost
from   remote_host import RemoteHost
import dircache

########################################################################
# Classes
########################################################################

def _get_host(context, variable):
    """Get a host instance according to a particular context variable.
    Return a default 'LocalHost' host if the variable is undefined.

    'context' -- The context to read the host descriptor from.

    'variable' -- The name to which the host descriptor is bound.

    returns -- A Host instance.

    """

    # This function is cloned from qm.test.classes.compilation_test !

    target_desc = context.get(variable)
    if target_desc is None:
        target = LocalHost({})
    else:
        f = lambda n: get_extension_class(n, "host", get_database())
        host_class, arguments = parse_descriptor(target_desc.strip(), f)
        target = host_class(arguments)
    return target

class CompilationTest(CTBase):

    execute = BooleanField(computed="true")

    def Run(self, context, result):
        """Determine from the context whether or not to execute
        the compiled executable."""
        self.execute = context.GetBoolean("CompilationTest.execute", True)
        super(CompilationTest, self).Run(context, result)


    def _GetEnvironment(self, context):

        env = os.environ.copy()
        paths = ':'.join(context.get('CompilerTest.library_dirs', '').split())
        if paths:
            ld_library_path = paths
            alf_library_path = '.:' + paths
            if 'LD_LIBRARY_PATH' in env:
                ld_library_path += ':' + env['LD_LIBRARY_PATH']
            env['LD_LIBRARY_PATH'] = ld_library_path
        return env

    def _GetTarget(self, context):
        """Return a target instance suitable to run the test executable.
        The choice of implementations is determined from the current context.

        The global context may have provided a 'CompilationTest.target' variable 
        to indicate the Host subclass to use. In addition, we consider the
        value of the 'ParallelActivator.num_processors' variable that may have
        been set by the ParallelActivator resource.

        If a value of num_processors > 1 is requested, we attempt to run the 
        executable using a command determined from the 'par_service.run' 
        variable. (This is typically 'mpirun'.)

        If the requested target is LocalHost, we replace it
        by CommandHost (which was designed precisely for such use-cases).

        If it is a CommandHost or any other host instance that is known
        to support injecting commands, we attempt to modify its arguments
        appropriately.

        Requesting num_processors > 1 with an unsupported host is an error
        and yields an exception."""




        # We can't use a regular import for LocalHost, as that wouldn't
        # match the type found via 'get_extension_class'.
        local_host = get_extension_class('local_host.LocalHost', 'host', get_database())
        target_desc = context.get('CompilationTest.target')
        if target_desc is None:
            host_class, arguments = local_host, {}
        else:
            f = lambda n: get_extension_class(n, 'host', get_database())
            host_class, arguments = parse_descriptor(target_desc.strip(), f)
        num_processors = context.get('ParallelActivator.num_processors', 1)

        if num_processors > 1:
            if host_class is local_host:
                host_class = CommandHost
                arguments['command'] = context.get('par_service.run')
                arguments['command_args'] = ['-np', str(num_processors)]
            elif host_class is CommandHost:
                # Assume that the command is required to invoke any
                # executable on this machine. Prepend the mpirun command
                # simply as an argument in that case.
                arguments['command_args'] = [context.get('par_service.run'),
                                             '-np', str(num_processors)]

            elif host_class is RemoteHost:
                pass # ...
            else:
                raise Exception('target "%s" does not support parallel execution'%target_desc)

            path = context.get('par_service.run')

        return host_class(arguments)


    def _CheckOutput(self, context, result, prefix, output, diagnostics):
        """Determine from the context whether or not to treat warnings
        as errors."""

        if output:
            result[prefix + "output"] = result.Quote(output)

        check_warnings = context.GetBoolean("CompilationTest.check_warnings", False)
        if not output or not check_warnings:
            return True

        lang = self.language
        compiler = context['CompilerTable.compilers'][lang]

        errors_occured = False
        diagnostics = compiler.ParseOutput(output)
        for d in diagnostics:
            # We only check for warnings, since errors are already dealt with
            # elsewhere.
            if d.severity == 'warning':
                errors_occured = True
                result.Fail("The compiler issued an un-expected warning.")

        return not errors_occured



class CompiledResource(Resource):
    """A CompiledResource fetches compilation parameters from environment
    variables CPPFLAGS, <lang>_options, and <lang>_ldflags in addition
    to the CompilerTable-related parameters."""

    options = SetField(TextField(), computed="true")
    ldflags = SetField(TextField(), computed="true")
    source_files = SetField(TextField(), computed="true")
    executable = TextField(computed="true")
    language = TextField()

    def SetUp(self, context, result):

        self._context = context
        self._compiler = CTBase({'options':self.options,
                                 'ldflags':self.ldflags,
                                 'source_files':self.source_files,
                                 'executable':self.executable,
                                 'language':self.language,
                                 'execute':False},
                                qmtest_id = self.GetId(),
                                qmtest_database = self.GetDatabase())
        
        self._compiler.Run(context, result)
        directory = self._compiler._GetDirectory(context)
        self._executable = os.path.join(directory, self.executable)
        context['CompiledResource.executable'] = self._executable
        

    def CleanUp(self, result):

        self._compiler._RemoveDirectory(self._context, result)


class DataTest(Test):
    """A DataTest runs an executable from a CompiledResource, with a data-file as input.
    """

    data_file = TextField(description="Arguments to pass to the executable.")

    def Run(self, context, result):

        executable = context['CompiledResource.executable']
        host = _get_host(context, 'CompilationTest.target')

        env = os.environ.copy()
        paths = ':'.join(context.get('CompilerTest.library_dirs', '').split())
        if paths:
            ld_library_path = paths
            if 'LD_LIBRARY_PATH' in env:
                ld_library_path += ':' + env['LD_LIBRARY_PATH']
            env['LD_LIBRARY_PATH'] = ld_library_path
        remote_data_file = os.path.basename(self.data_file)
        host.UploadFile(self.data_file, remote_data_file)
        status, output = host.UploadAndRun(executable, [remote_data_file],
                                           environment = env)
        host.DeleteFile(remote_data_file)
        if not result.CheckExitStatus('DataTest.', 'Program', status):
            result.Fail('Unexpected exit_code')        
        if output:
            result['DataTest.output'] = result.Quote(output)

class ParallelService(Resource):

    def SetUp(self, context, result):

        setup = Executable()
        command = []
        self.halt_command = []
        
        command = context.get('par_service.boot', '').split()
        self.halt_command = context.get('par_service.halt', '').split()

        if command:
            status = setup.Run(command)
            result.CheckExitStatus('ParallelService', ' '.join(command), status)

    def CleanUp(self, result):

        if self.halt_command:
            command = self.halt_command
            cleanup = Executable()
            status = cleanup.Run(command)
            result.CheckExitStatus('ParallelService', ' '.join(command), status)
        
            
class ParallelActivator(Resource):
    """This resource defines the 'ParallelActivator.use_num_processors'
    context variable to indicate that any dependent test should be run in parallel."""

    def SetUp(self, context, result):

        num_processors = context.get('par_service.num_processors')
        if num_processors:
            context['ParallelActivator.num_processors'] = int(num_processors)

class Database(CompilationTestDatabase):
    """'Database' stores the OpenVSIP test and benchmark suites.

    In addition to the CompilationTestDatabase behavior, we must:

    * make all tests depend on the ParallelService resource
    * add special handling for directories containing 'data/' subdirs.


    """
    no_exclusions = BooleanField()
    flags = DictionaryField(TextField(), TextField())
    excluded_subdirs = SetField(TextField(),
                                default_value = ['QMTest', 'data', 'build'],
                                description="Subdirectories not to scan for tests.",
                                computed="true")

    def __init__(self, *args, **kwds):

        super(Database, self).__init__(*args, **kwds)

        self.test_extensions['.py'] = 'python'

        if self.no_exclusions == 'false':
            if self.flags.get('have_mpi') != '1':
                self.excluded_subdirs.append('mpi')
                self.excluded_subdirs.append('parallel')
            if self.flags.get('enable_threading') != '1':
                self.excluded_subdirs.append('thread')
            if self.flags.get('have_ipp') != '1':
                self.excluded_subdirs.append('ipp')
            if self.flags.get('have_sal') != '1':
                self.excluded_subdirs.append('sal')
            if self.flags.get('have_fftw') != '1':
                self.excluded_subdirs.append('fftw')
            if self.flags.get('have_cuda') != '1':
                self.excluded_subdirs.append('cuda')
            if self.flags.get('enable_cvsip_bindings') != 'yes':
                self.excluded_subdirs.append('cvsip')
            if self.flags.get('enable_python_bindings') != 'yes':
                self.excluded_subdirs.append('python')
            if self.flags.get('enable_threading') != '1':
                self.excluded_subdirs.append('threading')

    def GetSubdirectories(self, directory):

        subdirs = super(Database, self).GetSubdirectories(directory)
        subdirs = [s for s in subdirs
                   if self.JoinLabels(directory, s) not in self.excluded_subdirs]
        return subdirs


    def GetIds(self, kind, directory = '', scan_subdirs = 1):
        # Directories containing 'data/' subdir are special.
        # Everything else is handled by the base class.

        dirname = os.path.join(self.srcdir, directory)
        if not os.path.isdir(dirname):
            raise NoSuchSuiteError, directory
        elif os.path.isdir(os.path.join(dirname, 'data')):
            if kind == Database.TEST:
                return [self.JoinLabels(directory, f)
                        for f in dircache.listdir(os.path.join(dirname, 'data'))
                        if f not in self.excluded_subdirs]
            else:
                return []
        else:
            return super(Database, self).GetIds(kind, directory, scan_subdirs)


    def GetExtension(self, id):

        if not id:
            return DirectorySuite(self, id)
            
        elif id == 'compiler_table':
            return CompilerTable({}, qmtest_id = id, qmtest_database = self)

        elif id == 'parallel_service':
            return ParallelService({}, qmtest_id = id, qmtest_database = self)
        elif id == 'parallel_activator':
            return ParallelActivator({}, qmtest_id = id, qmtest_database = self)

        resources = ['compiler_table', 'parallel_service']

        id_components = self.GetLabelComponents(id)
        # 'data' subdirectories have special meaning, and so
        # are not allowed as label components.
        if 'data' in id_components:
            return None

        dirname = os.path.join(self.srcdir, *id_components[:-1])
        basename = id_components[-1]

        file_ext = os.path.splitext(basename)[1]

        # If <dirname>/data is an existing directory...
        if os.path.isdir(os.path.join(dirname, 'data')):

            if file_ext in self.test_extensions:

                executable = os.path.splitext(os.path.basename(id))[0]
                if sys.platform == 'win32':
                    executable += '.exe'

                # ...<dirname>/<basename> is a resource.
                src = os.path.abspath(os.path.join(self.srcdir, id))
                return self._MakeTest(id,
                                      CompiledResource,
                                      language=self.test_extensions[file_ext],
                                      source_files=[src],
                                      executable=executable,
                                      resources=resources)
            else:
                # ...<dirname>/<basename> is a test.
                path = os.path.join(dirname, 'data', basename)
                if not os.path.isfile(path):
                    return None

                src = [f for f in dircache.listdir(dirname)
                       if os.path.splitext(f)[1] in self.test_extensions]
                # There must be exactly one source file, which
                # is our resource.
                if len(src) > 1:
                    raise DatabaseError('multiple source files found in %s'%dirname)

                resources.append(self.JoinLabels(*(id_components[:-1] + src)))
                return self._MakeTest(id,
                                      DataTest,
                                      resources=resources,
                                      data_file=path)
            

        src = os.path.join(self.srcdir, id)
        if file_ext in self.test_extensions and os.path.isfile(src):
            if file_ext == '.py':
                return self._MakePythonTest(id, src)
            else:
                executable = os.path.splitext(os.path.basename(id))[0]
                if sys.platform == 'win32':
                    executable += '.exe'

                # all tests in parallel/ should be run in parallel.
                if id_components[0] in ('mpi', 'parallel'):
                    resources.append('parallel_activator')

                return self._MakeTest(id,
                                      CompilationTest,
                                      language=self.test_extensions[file_ext],
                                      source_files=[src],
                                      executable=executable,
                                      resources=resources)
            
        elif os.path.isfile(src + '.qms'):
            qms = src + '.qms'
            # Expose the flags to the suite file so it can exclude ids
            # the same way the database itself does in the constructor.
            context = dict(flags=self.flags,
                           excluded_subdirs=self.excluded_subdirs)
            try:
                content = open(qms).read()
                exec content in context
            except:
                print 'Error parsing', qms
            test_ids=context.get('test_ids', [])
            suite_ids=context.get('suite_ids', [])
            return ExplicitSuite(is_implicit=False,
                                 test_ids=test_ids, suite_ids=suite_ids,
                                 qmtest_id = id, qmtest_database = self)

        elif os.path.isdir(src):
            if not basename in self.excluded_subdirs:
                return DirectorySuite(self, id)

        else:
            return None


    def _MakeTest(self, id, class_, **args):

        return class_(args, qmtest_id = id, qmtest_database = self)

    def _MakePythonTest(self, id, src):

        source = '\n'.join(open(src, 'r').readlines())
        return PythonExecTest(source=source, qmtest_id = id, qmtest_database = self)
