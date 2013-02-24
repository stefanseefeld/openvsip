var isNav4 = false, isIE4 = false;
if (parseInt(navigator.appVersion.charAt(0)) == 4)
{
  isNav4 = (navigator.appName == "Netscape") ? true : false;
}
else if (parseInt(navigator.appVersion.charAt(0)) >= 4) 
{
  isIE4 = (navigator.appName.indexOf("Microsoft") != -1) ? true : false;
}
var isMoz = (isNav4 || isIE4) ? false : true;

showImage = new Image(); hideImage = new Image();

/* The 'className' member only returns the entire (whitespace-separated) list
   of classes, while CSS selectors use individual classes. has_class allows to
   check for an individual class within such a list.*/
function has_class(object, class_name)
{
  if (!object.className) return false;
  return (object.className.search('(^|\\s)' + class_name + '(\\s|$)') != -1);
}

function get_children_with_class(parent, class)
{
  var result = new Array();
  for (var i = 0; i < parent.childNodes.length; ++i)
  {
    if (has_class(parent.childNodes[i], class))
      result[result.length] = parent.childNodes[i];
  }
  return result;
}

function get_child_with_class(parent, class)
{
  var children = get_children_with_class(parent, class);
  if (children.length != 0) return children[0];
  else return false; /*??*/
}

function tree_init(show_src, hide_src) 
{
  showImage.src = show_src; hideImage.src = hide_src;
}
function tree_node_toggle(id) 
{
  if (isMoz)
  {
    section = document.getElementById(id);
    image = document.getElementById(id+"_img");
    if (section.style.display == "none") 
    {
      section.style.display = "";
      image.src = showImage.src;
    }
    else
    {
      section.style.display = "none";
      image.src = hideImage.src;
    }
  }
  else if (isIE4)
  {
    section = document.items[id];
    image = document.images[id+"_img"];
    if (section.style.display == "none") 
    {
      section.style.display = "";
      image.src = showImage.src;
    }
    else
    {
      section.style.display = "none";
      image.src = hideImage.src;
    }
  }
  else if (isNav4) 
  {
    section = document.items[id];
    image = document.images[id+"_img"];
    if (section.display == "none") 
    {
      section.style.display = "";
      image.src = showImage.src;
    }
    else
    {
      section.display = "none";
      image.src = hideImage.src;
    }
  }
}
var tree_max_node = 0;
function tree_open_all()
{
  for (i = 1; i <= tree_max_node; i++)
  {
    id = "tree"+i;
    section = document.getElementById(id);
    image = document.getElementById(id+"_img");
    section.style.display = "";
    image.src = showImage.src;
  }
}
function tree_close_all()
{
  for (i = 1; i <= tree_max_node; i++)
  {
    id = "tree"+i;
    section = document.getElementById(id);
    image = document.getElementById(id+"_img");
    section.style.display = "none";
    image.src = hideImage.src;
  }
}

tree_init("tree_open.png", "tree_close.png");

function go(frame1, url1, frame2, url2)
{
  window.parent.frames[frame1].location=url1;
  window.parent.frames[frame2].location=url2;
  return false;
}

/* A body section is a section inside a Body part. */

function body_section_expand(id) 
{
  section = document.getElementById(id);
  section.style.display = 'block';
  toggle = document.getElementById('toggle_' + id);
  toggle.firstChild.data = '-';
 }

function body_section_collapse(id) 
{
  section = document.getElementById(id);
  section.style.display = 'none';
  toggle = document.getElementById('toggle_' + id);
  toggle.firstChild.data = '+';
  toggle.style.display = 'inline';
}

function body_section_toggle(id) 
{
  section = document.getElementById(id);
  toggle = document.getElementById('toggle_' + id);
  if (section.style.display == 'none') 
  {
    section.style.display = 'block';
    toggle.firstChild.data = '-';
  }
  else
  {
    section.style.display = 'none';
    toggle.firstChild.data = '+';
  }
}

function body_section_collapse_all()
{
  divs = document.getElementsByTagName('div');
  for (var i = 0; i < divs.length; ++i)
  {
    if (has_class(divs[i], 'expanded') && divs[i].hasAttribute('id'))
    {
      body_section_collapse(divs[i].getAttribute('id'));
    }
  }
}

function body_section_init()
{
  divs = document.getElementsByTagName('div');
  for (var i = 0; i < divs.length; ++i)
  {
    if (has_class(divs[i], 'heading'))
    {
      toggle = get_child_with_class(divs[i], 'toggle');
      toggle.style.display='inline'
    }
  }
}

function decl_doc_expand(id) 
{
  doc = document.getElementById(id);
  collapsed = get_children_with_class(doc, 'collapsed');
  for (var i = 0; i < collapsed.length; ++i)
  {
    collapsed[i].style.display='none';
  }
  expanded = get_children_with_class(doc, 'expanded');
  for (var i = 0; i < expanded.length; ++i)
  {
    expanded[i].style.display='block';
  }
  collapse_toggle = get_child_with_class(doc, 'collapse-toggle');
  collapse_toggle.style.display='block';
}
function decl_doc_collapse(id) 
{
  doc = document.getElementById(id);
  expanded = get_children_with_class(doc, 'expanded');
  for (var i = 0; i < expanded.length; ++i)
  {
    expanded[i].style.display='none';
  }
  collapse_toggle = get_child_with_class(doc, 'collapse-toggle');
  collapse_toggle.style.display='none';
  collapsed = get_children_with_class(doc, 'collapsed');
  for (var i = 0; i < collapsed.length; ++i)
  {
    collapsed[i].style.display='block';
  }
}

function decl_doc_collapse_all()
{
  divs = document.getElementsByTagName('div');
  for (var i = 0; i < divs.length; ++i)
  {
    if (has_class(divs[i], 'collapsible'))
    {
      decl_doc_collapse(divs[i].getAttribute('id'));
    }
  }
}

function load()
{
  decl_doc_collapse_all();
  body_section_init();
}