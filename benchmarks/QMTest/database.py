########################################################################
#
# Copyright (c) 2009 by CodeSourcery, Inc.  All rights reserved. 
#
# Contents:
#   Test, Resource, and Database classes for the Sourcery VSIPL++ 
#   benchmark suite.
#
########################################################################

import os
import qm
import qm.test.base
from   qm.test.resource import Resource
from   qm.test.database import NoSuchTestError
from   qm.test.parameter_database import ParameterDatabase
from   qm.test.classes.explicit_suite import ExplicitSuite
from   qm.test.classes.compilation_test import ExecutableTest
from   qm.test.classes.compilation_test_database import *
from   qm.test.directory_suite import DirectorySuite
from   benchmark import Benchmark, BenchmarkExecutable
import dircache

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

# This database started as a clone of the SV++ test database.
# It may eventually be merged back, if we decide to run benchmarks as
# part of the regression tests.
class Database(CompilationTestDatabase):
    """'Database' stores the Sourcery VSIPL++ benchmark suite.

    In addition to the CompilationTestDatabase behavior, we must:

    * make all tests depend on the ParallelService resource
    * add special handling for directories containing 'data/' subdirs.


    """
    no_exclusions = BooleanField()
    flags = DictionaryField(TextField(), TextField())
    excluded_subdirs = SetField(TextField(),
                                default_value = ['QMTest', '.svn', 'data', 'build'],
                                description="Subdirectories not to scan for benchmarks.",
                                computed="true")

    def __init__(self, *args, **kwds):

        super(Database, self).__init__(*args, **kwds)

        if self.no_exclusions == 'false':
            if self.flags.get('VSIP_IMPL_HAVE_MPI') != '1':
                self.excluded_subdirs.append('mpi')
            if self.flags.get('VSIP_IMPL_HAVE_IPP') != '1':
                self.excluded_subdirs.append('ipp')
            if self.flags.get('VSIP_IMPL_HAVE_SAL') != '1':
                self.excluded_subdirs.append('sal')
            if self.flags.get('VSIP_IMPL_HAVE_LAPACK') != '1':
                self.excluded_subdirs.append('lapack')
            if self.flags.get('VSIP_IMPL_FFTW3') != '1':
                self.excluded_subdirs.append('fftw3')
            if self.flags.get('enable_cvsip_bindings') != 'yes':
                self.excluded_subdirs.append('cvsip')
            if self.flags.get('VSIP_IMPL_HAVE_CUDA') != '1':
                self.excluded_subdirs.append('cuda')
            if self.flags.get('VSIP_IMPL_HAVE_CBE_SDK') != '1':
                self.excluded_subdirs.append('cell')

    def GetSubdirectories(self, directory):

        subdirs = super(Database, self).GetSubdirectories(directory)
        subdirs = [s for s in subdirs
                   if self.JoinLabels(directory, s) not in self.excluded_subdirs]
        return subdirs


    def GetIds(self, kind, directory = '', scan_subdirs = 1):
        # The CompilationTestDatabase maps all source files to tests.
        # Here, we need to filter out main.cpp, which gets linked to everything else.

        dirname = os.path.join(self.srcdir, directory)
        if not os.path.isdir(dirname):
            raise NoSuchSuiteError, directory

        if kind in (Database.TEST, Database.RESOURCE):
            ids = [self.JoinLabels(directory, f)
                   for f in dircache.listdir(dirname)
                   if (os.path.isfile(os.path.join(dirname, f)) and
                       os.path.splitext(f)[1] in self.test_extensions and
                       f is not 'main.cpp')]
            # Ids with extensions stripped off are tests.
            if kind == Database.TEST:
                ids = [os.path.splitext(i)[0] for i in ids]
            
        else: # SUITE
            ids = [self.JoinLabels(directory, d)
                   for d in self.GetSubdirectories(directory)
                   if d not in self.excluded_subdirs]

        if scan_subdirs:
            for subdir in dircache.listdir(dirname):
                if (subdir not in self.excluded_subdirs
                    and os.path.isdir(os.path.join(dirname, subdir))):
                    dir = self.JoinLabels(directory, subdir)
                    ids.extend(self.GetIds(kind, dir, True))

        return ids
    

    def GetExtension(self, id):

        if not id:
            return DirectorySuite(self, id)
            
        elif id == 'compiler_table':
            return CompilerTable({}, qmtest_id = id, qmtest_database = self)

        elif id == 'parallel_service':
            return ParallelService({}, qmtest_id = id, qmtest_database = self)

        resources = [] #['compiler_table', 'parallel_service']

        id_components = self.GetLabelComponents(id)

        dirname = os.path.join(self.srcdir, *id_components[:-1])
        basename = id_components[-1]
        src = os.path.join(self.srcdir, id)
        file_ext = os.path.splitext(src)[1]

        # This may be a compiled resource
        if file_ext:
            if file_ext in self.test_extensions and os.path.isfile(src):
                return BenchmarkExecutable(qmtest_id=id, qmtest_database=self,
                                           src=src,
                                           executable=os.path.splitext(id)[0])

        # It may be a directory
        elif os.path.isdir(src):
            if not basename in self.excluded_subdirs:
                return DirectorySuite(self, id)

        # If there is a corresponding source, it's a test.
        elif os.path.isfile(src + '.cpp'):
            resources = [id + '.cpp']
            return self._MakeBenchmark(id,
                                       Benchmark,
                                       program='./%s'%id,
                                       resources=resources)
        else:
            return None


    def _MakeBenchmark(self, id, class_, **args):
        return class_(args, qmtest_id = id, qmtest_database = self)

class BenchmarkDatabase(ParameterDatabase):
    """The benchmark database provides benchmark tests as specified in
    a configuration file. It uses the ParameterDatabase to fold
    'implementations' with 'parameters'."""

    srcdir = TextField()
    no_exclusions = BooleanField()
    flags = DictionaryField(TextField(), TextField())

    def __init__(self, *args, **kwds):

        db = Database(*args, **kwds)
        args = [db] + list(args)
        ParameterDatabase.__init__(self, *args, **kwds)
        self._parameters = {}

    def _GetParametersForTest(self, test_id):

        if test_id not in self._parameters:
            try:
                db = self.GetWrappedDatabase()
                test = db.GetTest(test_id)
                # Each parametrized test has an associated BenchmarkExecutable
                # as resource...
                resource_id = test.GetArguments()['resources'][0]
                resource = db.GetResource(resource_id)
                bm = resource.GetResource()
                if isinstance(bm, BenchmarkExecutable):
                    # ...from which we extract the parameter set.
                    self._parameters[test_id] = bm.GetCases()
            except (NoSuchTestError, KeyError, IOError):
                pass
        return self._parameters.get(test_id, {}).keys()

    def _GetArgumentsForParameter(self, test_id, parameter):

        return dict(case=self._parameters[test_id][parameter])


    def HasSuite(self, suite_id):

        qms = os.path.join(self.srcdir, suite_id) + '.qms'
        return (os.path.isfile(qms) or
                ParameterDatabase.HasSuite(self, suite_id))
        
    def GetSuite(self, suite_id):

        qms = os.path.join(self.srcdir, suite_id) + '.qms'
        if os.path.isfile(qms):
            wd = self.GetWrappedDatabase()
            # Expose the flags to the suite file so it can exclude ids
            # the same way the Database itself does in the constructor.
            context = dict(flags=wd.flags,
                           excluded_subdirs=wd.excluded_subdirs)
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
        else:
            return ParameterDatabase.GetSuite(self, suite_id)
