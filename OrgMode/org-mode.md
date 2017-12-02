<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. 笔记结构</a>
<ul>
<li><a href="#sec-1-1">1.1. 章节</a></li>
<li><a href="#sec-1-2">1.2. 列表</a></li>
<li><a href="#sec-1-3">1.3. 脚注</a></li>
<li><a href="#sec-1-4">1.4. 表格</a></li>
<li><a href="#sec-1-5">1.5. 链接</a></li>
<li><a href="#sec-1-6">1.6. 待办事项(TODO)</a></li>
</ul>
</li>
<li><a href="#sec-2">2. 标记</a>
<ul>
<li><a href="#sec-2-1">2.1. 标签Tags</a></li>
<li><a href="#sec-2-2">2.2. 时间</a></li>
<li><a href="#sec-2-3">2.3. 特殊文本格式</a></li>
<li><a href="#sec-2-4">2.4. 富文本导出</a>
<ul>
<li><a href="#sec-2-4-1">2.4.1. 设置标题和目录</a></li>
<li><a href="#sec-2-4-2">2.4.2. 添加引用</a></li>
<li><a href="#sec-2-4-3">2.4.3. 设置居中</a></li>
<li><a href="#sec-2-4-4">2.4.4. 设置样例</a></li>
<li><a href="#sec-2-4-5">2.4.5. 注释</a></li>
<li><a href="#sec-2-4-6">2.4.6. LATEX</a></li>
<li><a href="#sec-2-4-7">2.4.7. 源代码</a></li>
<li><a href="#sec-2-4-8">2.4.8. 文章信息</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#sec-3">3. 导出</a>
<ul>
<li><a href="#sec-3-1">3.1. PDF</a></li>
</ul>
</li>
<li><a href="#sec-4">4. 参考资料</a></li>
</ul>
</div>
</div>


# 笔记结构<a id="sec-1" name="sec-1"></a>

## 章节<a id="sec-1-1" name="sec-1-1"></a>

在orgmode中，n个\*表示n级章节，例如：

    * 一级标题
    ** 二级标题

快捷键：  
S-Tab 展开，折叠所有章节  
Tab 对当前章节进行折叠  
M-left/right 升级，降级标题  

## 列表<a id="sec-1-2" name="sec-1-2"></a>

列表可以列出所有待完成事项。orgmode可以加入checkbok "[ ]" 来标记任务的完成情况。  
如果一个任务有多个子任务，还可以根据子任务的完成情况来计算总进度，只需要在总任务下面加上“[ % ]”或者“[ / ]”（不包含空格）。  
列表分为两种，有序列表以“1.”或者“1)”开头，无序列表以“+”或者“-”开头。同样，后面要跟一个空格。  

    + root
      + branch1
      + branch2
    
    1) [ ] root1 [%]
      1) [ ] branch1
      2) [ ] branch2
    2) [ ] root2

快捷键：  
M-RET 插入统计列表项  
M-S-RET 插入一个带checkbox的列表项  
C-c C-c 改变checkbox的状态  
M-left/right 改变列表项的层级关系  
M-up/down 上下移动列表项  

## 脚注<a id="sec-1-3" name="sec-1-3"></a>

    插入脚注采用[fn:1]的方式，在最下面插入[fn:1]OrgMode-Note。这个标签是可以点击的。

## 表格<a id="sec-1-4" name="sec-1-4"></a>

orgmode提供的方便的表格操作。最独特的是支持类似Excel的表格函数，可以完成求和等操作。  
创建表格时，首先输入表头：

    | Name | Phone | sub1 | sub2 | total|
    |-

然后按下tab，表格就会自动生成。然后可以输入数据，再输入的时候，按tab可以跳到右方表格，按enter能跳到下方表格。按住shift则反向跳。输入完成后，按C-c C-c可以对齐表格。

    | Name |  Phone | sub1 | sub2 | total |
    |------+--------+------+------+-------|
    | Tom  | 134... |   89 |   98 |       |
    | Jack | 152... |   78 |   65 |       |
    | Ken  | 123... |   76 |   87 |       |
    | Ana  | 157... |   87 |   78 |       |

对于表格函数，我们在total列选择一个位置，然后输入

    =$3+$4

按下C-u C-c C-c，orgmode就能计算所有的第三列加第四列的和，并放到第五列。

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="left" />

<col  class="left" />

<col  class="right" />

<col  class="right" />

<col  class="right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="left">Name</th>
<th scope="col" class="left">Phone</th>
<th scope="col" class="right">sub1</th>
<th scope="col" class="right">sub2</th>
<th scope="col" class="right">total</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">Tom</td>
<td class="left">134&#x2026;</td>
<td class="right">89</td>
<td class="right">98</td>
<td class="right">187</td>
</tr>


<tr>
<td class="left">Jack</td>
<td class="left">152&#x2026;</td>
<td class="right">78</td>
<td class="right">65</td>
<td class="right">143</td>
</tr>


<tr>
<td class="left">Ken</td>
<td class="left">123&#x2026;</td>
<td class="right">76</td>
<td class="right">87</td>
<td class="right">163</td>
</tr>


<tr>
<td class="left">Ana</td>
<td class="left">157&#x2026;</td>
<td class="right">87</td>
<td class="right">78</td>
<td class="right">165</td>
</tr>
</tbody>
</table>

如果要插入行和列，可以在表头添加一个标签或者新起一行，再输入|调整格式即可。其中最后一行是orgmode自动添加的。  
快捷键：  
C-c | 通过输入大小的方式插入表格  
C-c C-c 对齐表格  
tab 跳到右边一个表格   
enter 跳到下方的表格   
M-up/right/left/right 上下左右移动行（列）  
M-S-up/right/left/right 向上下左右插入行（列）  

## 链接<a id="sec-1-5" name="sec-1-5"></a>

用于链接一些资源地址，比如文件，图片，URL等。链接的格式是：

    [[链接地址][链接内容]]
    
    如：
    [[http://orgmode.org/orgguide.pdf][grgguid.pdf]]]
    [[file:/home/maple/图片/test.jpg][a picture]]
    
    如果去掉标签，则能直接显示图片：
    [[file:/home/maple/图片/test.jpg]]

直接显示的图片在Emacs默认不显示，需要按C-c C-x C-v才能显示。  
快捷键：  
C-c C-x C-v 预览图片  

## 待办事项(TODO)<a id="sec-1-6" name="sec-1-6"></a>

TODO是orgmode最有特色的一个功能，可以完成一个GDT。  
TODO也是一类标题，因此也需要“\*”开头。

    ** TODO 刷题

初始TODO为红色，如果将光标移动到该行并按下C-c C-t，则发现TODO变成了DONE。

    *** TODO [# A] 任务1
    *** TODO [# B] 任务2
    *** TODO 总任务 [33%]
    **** TODO 子任务1
    **** TODO 子任务2 [0%]
         - [-] subsub1 [1/2]
          - [ ] subsub2
          - [X] subsub3
    **** DONE 一个已完成的任务

快捷键：  
C-c C-t 变换TODO的状态   
C-c / t 以树的形式展示所有的 TODO   
C-c C-c 改变 checkbox状态   
C-c , 设置优先级（方括号里的ABC）   
M-S-RET 插入同级TODO标签  

# 标记<a id="sec-2" name="sec-2"></a>

## 标签Tags<a id="sec-2-1" name="sec-2-1"></a>

在orgmode中，可以给每一章添加一个标签。可以通过树的结构来查看带标签的章节。在每一节中，子标题的标签会继承父标题的标签。

    *** 章标题                                                       :work:learn:
    **** 节标题1                                                      :fly:plane:
    **** 节标题2                                                        :car:run:

快捷键：  
C-c C-q 为标题添加标签   
C-c / m 生成带标签的树   

## 时间<a id="sec-2-2" name="sec-2-2"></a>

orgmode可以利用emacs的时间插入当前的时间。输入C-c . 会出现一个日历，选择相应的时间插入即可。<span class="timestamp-wrapper"><span class="timestamp">&lt;2017-12-01 Fri&gt;</span></span>   
时间可以添加DEADLINE和SCHEDULED表示时间的类型。

    DEADLINE:<2017-12-01 Fri>

快捷键：   
C-c . 插入时间   

## 特殊文本格式<a id="sec-2-3" name="sec-2-3"></a>

    *bold*
    /italic/
    _underlined_
    =code=
    ~verbatim~
    +strike-through+

## 富文本导出<a id="sec-2-4" name="sec-2-4"></a>

主要用于导出pdf或者html时制定导出选项。   

### 设置标题和目录<a id="sec-2-4-1" name="sec-2-4-1"></a>

    # +TITLE: This is the title of the document
    # +OPTIONS: toc:2 (只显示两级目录)
    # +OPTIONS: toc:nil (不显示目录)

### 添加引用<a id="sec-2-4-2" name="sec-2-4-2"></a>

    #+BEGIN_QUOTE
    Everything should be made as simple as possible,
    but not any simpler -- Albert Einstein
    #+END_QUOTE

> Everything should be made as simple as possible,
> but not any simpler &#x2013; Albert Einstein

快捷键：  
键入 <q 之后按下 tab 自动补全。

### 设置居中<a id="sec-2-4-3" name="sec-2-4-3"></a>

    #+BEGIN_CENTER
    Everything should be made as simple as possible,but not any simpler
    #+END_CENTER

<div class="center">
Everything should be made as simple as possible,but not any simpler
</div>

快捷键：  
键入 <c 之后按下 tab 自动补全。

### 设置样例<a id="sec-2-4-4" name="sec-2-4-4"></a>

    实际应该为BEGIN_EXAMPLE和END_EXAMPLE
    
    #+BEGINEXAMPLE
    这里面的字符不会被转义。
    #+ENDEXAMPLE

快捷键：  
键入 <e 之后按下 tab 自动补全。

### 注释<a id="sec-2-4-5" name="sec-2-4-5"></a>

    # comment
    或者：
    #+BEGIN_COMMENT
    这里的注释不会被导出
    #+END_COMMENT

### LATEX<a id="sec-2-4-6" name="sec-2-4-6"></a>

    嵌入公式：\( \) 或 $ $
    行间公式：\[ \] 或 $$ $$

LATEX能支持直接输入LATEX。

$$ ax+by+c $$
快捷键：  
C-c C-x C-l 预览LATEX图片。  

### 源代码<a id="sec-2-4-7" name="sec-2-4-7"></a>

    #+BEGIN_SRC C++
    #include <cstdio>
    using namespace std;
    int main() {
      int a=1;
      int b=1;
      printf("%d\n",a+b);
    }
    
    #+END_SRC

    #include <cstdio>
    using namespace std;
    int main() {
      int a=1;
      int b=1;
      printf("%d\n",a+b);
    }

快捷键：  
C-c C-c 对当前代码块求值

### 文章信息<a id="sec-2-4-8" name="sec-2-4-8"></a>

    #+TITLE:       the title to be shown (default is the buffer name)
    #+AUTHOR:      the author (default taken from user-full-name)
    #+DATE:        a date, fixed, of a format string for format-time-string
    #+EMAIL:       his/her email address (default from user-mail-address)
    #+DESCRIPTION: the page description, e.g. for the XHTML meta tag
    #+KEYWORDS:    the page keywords, e.g. for the XHTML meta tag
    #+LANGUAGE:    language for HTML, e.g. ‘en’ (org-export-default-language)
    #+TEXT:        Some descriptive text to be inserted at the beginning.
    #+TEXT:        Several lines may be given.
    #+OPTIONS:     H:2 num:t toc:t \n:nil @:t ::t |:t ^:t f:t TeX:t ...
    #+LINK_UP:     the ``up'' link of an exported page
    #+LINK_HOME:   the ``home'' link of an exported page
    #+LATEX_HEADER: extra line(s) for the LaTeX header, like \usepackage{xyz}

# 导出<a id="sec-3" name="sec-3"></a>

C-c C-e 选择导出样式。

## PDF<a id="sec-3-1" name="sec-3-1"></a>

PDF格式的导出需要先导出为LATEX文件然后再编译为PDF文件。  
想要设置导出的页面大小的话，需要修改

    \documentclass[a4paper]{article}

如果PDF文件中含有中文，需要更改编译器为Xelatex然后在头文件中的\\documentclass下方加入  

    \usepackage{xeCJK}
    \setCJKmainfont{宋体}

如果需要插入代码，需要在头文件中加入  

    \usepackage{minted}

页边距的设置：

    \usepackage{geometry}
    \geometry{left=2.0cm,right=2.0cm,top=2.5cm,bottom=2.5cm}

去掉目录红边：

    \hypersetup{colorlinks=true,linkcolor=red}

字体设置：

    \setmainfont{Times New Roman}
    \setsansfont{Arial}
    \setmonofont{Courier New}

# 参考资料<a id="sec-4" name="sec-4"></a>

[Org-Manual 7.8](https://github.com/marboo/orgmode-cn/blob/master/org.org)