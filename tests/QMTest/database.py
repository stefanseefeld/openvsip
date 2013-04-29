########################################################################
#
# File:   database.py
# Author: Stefan Seefeld
# Date:   2008-05-03
#
# Contents:
#   Test, Resource, and Database classes for the Sourcery VSIPL++ test suite.
#
# Copyright (c) 2008 by CodeSourcery, Inc.  All rights reserved. 
#
########################################################################

########################################################################
# Imports
########################################################################

import fnmatch
import os
import qm
import qm.test.base
from   qm.fields import *
from   qm.executable import *
from   qm.test.resource import Resource
from   qm.test.classes.explicit_suite import ExplicitSuite
from   qm.test.classes.compilation_test import ExecutableTest
from   qm.test.classes.compilation_test_database import CompilationTestDatabase
from   qm.test.classes.compilation_test_database import CompilationTest as CTBase
from   qm.test.classes.compiler_test import CompilerTest
from   qm.test.classes.compiler_table import CompilerTable
from   qm.test.classes.python import ExecTest as PythonExecTest
from   qm.test.directory_suite import DirectorySuite
import dircache

########################################################################
# Classes
########################################################################

class CompilationTest(CTBase):

    execute = BooleanField(computed="true")

    def Run(self, context, result):
        """Determine from the context whether or not to execute
        the compiled executable."""

        self.execute = context.GetBoolean("CompilationTest.execute", True)
        super(CompilationTest, self).Run(context, result)


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


class ParallelService(Resource):

    def SetUp(self, context, result):

        setup = Executable()
        command = []
        self.halt_command = []
        
        service = context.get('par_service.name')
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
        
            
class Database(CompilationTestDatabase):
    """'Database' stores the Sourcery VSIPL++ test suite.

    In addition to the CompilationTestDatabase behavior, we must:

    * make all tests depend on the ParallelService resource
    * add special handling for directories containing 'data/' subdirs.


    """
    no_exclusions = BooleanField()
    flags = DictionaryField(TextField(), TextField())
    excluded_subdirs = SetField(TextField(),
                                default_value = ['QMTest', '.svn', 'data', 'build'],
                                description="Subdirectories not to scan for tests.",
                                computed="true")

    def __init__(self, *args, **kwds):

        super(Database, self).__init__(*args, **kwds)

        self.test_extensions['.py'] = 'python'

        if self.no_exclusions == 'false':
            if self.flags.get('enable_cvsip_bindings') != 'yes':
                self.excluded_subdirs.append('cvsip')
            if self.flags.get('enable_python_bindings') != 'yes':
                self.excluded_subdirs.append('python')
            if self.flags.get('VSIP_IMPL_HAVE_CUDA') != '1':
                self.excluded_subdirs.append('cuda')
            if self.flags.get('VSIP_IMPL_HAVE_CBE_SDK') != '1':
                self.excluded_subdirs.append('ukernel')
                self.excluded_subdirs.append('cbe')
            if self.flags.get('VSIP_IMPL_ENABLE_THREADING') != '1':
                self.excluded_subdirs.append('thread')

    def GetSubdirectories(self, directory):

        subdirs = super(Database, self).GetSubdirectories(directory)
        subdirs = [s for s in subdirs
                   if self.JoinLabels(directory, s) not in self.excluded_subdirs]
        return subdirs


    def GetIds(self, kind, directory = '', scan_subdirs = 1):
        # Directories containing 'data/' subdir are special.
        # Everything else is handled by the base class.

        dirname = os.path.join(self.srcdir, directory)
        if os.path.isdir(os.path.join(dirname, 'data')):
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
                                      ExecutableTest,
                                      resources=resources,
                                      args=[path])
            

        src = os.path.join(self.srcdir, id)
        if file_ext in self.test_extensions and os.path.isfile(src):
            if file_ext == '.py':
                return self._MakePythonTest(id, src)
            else:
                executable = os.path.splitext(os.path.basename(id))[0]
                if sys.platform == 'win32':
                    executable += '.exe'

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
