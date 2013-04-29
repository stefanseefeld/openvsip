#
# Copyright (c) 2005 CodeSourcery
# Copyright (c) 2013 Stefan Seefeld
# All rights reserved.
#
# This file is part of OpenVSIP. It is made available under the
# license contained in the accompanying LICENSE.GPL file.

from qm.fields import *
from qm.executable import *
from qm.extension import parse_descriptor
from qm.test.base import get_extension_class
from qm.test.test import Test
from qm.test.resource import Resource
from qm.test.database import get_database
from docutils.core import *
from docutils.nodes import *
from StringIO import StringIO

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

class InfoExtractor(SparseNodeVisitor):

    def __init__(self, document):

        SparseNodeVisitor.__init__(self, document)
        self._active = False
        self.description = ''
        self.cases = {}
        
    def visit_paragraph(self, node):
    
        if node.astext() == 'Description:':
            self.description = node.next_node(descend=0, siblings=1).astext()
        elif node.astext() == 'Cases:':
            self._active = True

    def visit_definition_list_item(self, node):

        if self._active:
            term, category, definition = node[0].astext(), node[1], node[-1]
            if category.tagname != 'classifier':
                raise Exception('missing option for "%s"'%term)
            if term in self.cases.keys():
                raise Exception('duplicate case "%s"'%term)
            self.cases[term] = (category.astext().split(),
                                '%s, %s'%(self.description, definition.astext()))
        else:
            return SparseNodeVisitor.visit_definition_list_item(self, node)



class BenchmarkExecutable(Resource):
    """A BenchmarkExecutable compiles a benchmark so it can be ran
    by the Benchmark."""

    executable = TextField(computed="true")
    src = TextField(computed="true")

    def GetCases(self):
        """Report benchmark cases for this benchmark resource.
        Cases may be embedded in a source file preamble (a comment block
        prefixed with '///', and using ReST markup).
        By default, a 'default' case is reported for option '-1'."""

        src = open(self.src).readlines()
        # Find the first block of '///' comment.
        while src and not src[0].startswith('///'):
            del src[0]
        if not src:
            return {'default': (['-1'], '')}
        i = 0
        while src and src[i].startswith('///'):
            i += 1
        src = src[:i]
        block = ''.join([line[3:] for line in src])

        errstream = StringIO()
        settings = {}
        settings['halt_level'] = 2
        settings['warning_stream'] = errstream
        try:
            doctree = publish_doctree(block, settings_overrides=settings)
        except docutils.utils.SystemMessage, error:
            xx, line, message = str(error).split(':', 2)
            msg = '%s:%s'%(self.src, message)
            raise qm.QMException(msg)
        extractor = InfoExtractor(doctree)
        doctree.walk(extractor)
        return extractor.cases or {'default': (['-1'], '')}


    def SetUp(self, context, result):
        
        cmd = ['make', self.executable]
        make = RedirectedExecutable()
        status = make.Run(cmd)
        if make.stdout:
            result['BenchmarkExecutable.compilation_output'] = result.Quote(make.stdout)
        if make.stderr:
            result['BenchmarkExecutable.compilation_error'] = result.Quote(make.stderr)
        if not result.CheckExitStatus('BenchmarkExecutable.compilation_', ' '.join(cmd), status):
            return
        context['BenchmarkExecutable.executable'] = self.executable

    def CleanUp(self, result): pass


class Benchmark(Test):
    
    arguments = SetField(TextField())
    description = TextField(computed="true")

    def __init__(self, *args, **kwds):
        super(Benchmark, self).__init__(*args, **kwds)
        self.arguments = self.arguments + ['-data', '-samples', '6', '-fix_loop']

    def Run(self, context, result):

        executable = context['BenchmarkExecutable.executable']
        host = _get_host(context, 'CompilationTest.target')

        env = os.environ.copy()
        paths = ':'.join(context.get('CompilerTest.library_dirs', '').split())
        if paths:
            ld_library_path = paths
            alf_library_path = paths
            if 'LD_LIBRARY_PATH' in env:
                ld_library_path += ':' + env['LD_LIBRARY_PATH']
            if 'ALF_LIBRARY_PATH' in env:
                alf_library_path += ':' + env['ALF_LIBRARY_PATH']
            env['LD_LIBRARY_PATH'] = ld_library_path
            env['ALF_LIBRARY_PATH'] = alf_library_path
        status, output = host.UploadAndRun(os.path.join('.', executable), self.arguments, environment = env)
        if not result.CheckExitStatus('Benchmark.', 'Program', status):
            result.Fail('Unexpected exit_code')        
        if output:
            result['Benchmark.output'] = result.Quote(output)

