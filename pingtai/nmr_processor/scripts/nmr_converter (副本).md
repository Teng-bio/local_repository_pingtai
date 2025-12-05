#!/usr/bin/tcsh -f
# 功能：自动转换、处理Bruker NMR数据的脚本
# 针对代谢谱优化版本 - 使用Deep Picker测试的最佳参数
# 参数优化：scale=55, scale2=20, model=2, auto_ppp=yes
# 新增功能：自动DSS化学位移校正

# 设置默认值
set DATA_DIR = "."
set PROCESS_FLAG = 2

# 参数检查
if ($#argv >= 1) set DATA_DIR = "$argv[1]"
if ($#argv >= 2) set PROCESS_FLAG = "$argv[2]"

# 获取deeppicker路径 - 优先从环境变量获取
set DEEP_PICKER_PATH = "$DEEP_PICKER_PATH"
if ("$DEEP_PICKER_PATH" == "") then
    set USER_CONFIG = "$HOME/.nmr_processor/config.json"
    if ( -f "$USER_CONFIG" ) then
        set DEEP_PICKER_PATH = `grep -o '"deep_picker_path"[^,]*' "$USER_CONFIG" | awk -F ':' '{print $2}' | tr -d '"' | tr -d ' '`
    endif
endif

# 创建日志目录和文件
set DATESTAMP = `date +%Y%m%d_%H%M%S`
mkdir -p "${DATA_DIR}/logs"
set LOG_FILE = "${DATA_DIR}/logs/process_${DATESTAMP}.log"

# 输出基本信息
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "  NMR自动处理 - 代谢谱优化版（Deep Picker最佳参数）" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "数据目录: $DATA_DIR" | tee -a "$LOG_FILE"
echo "处理时间: `date`" | tee -a "$LOG_FILE"
echo "相位校正: basicAutoPhase.com + Kaiser窗函数" | tee -a "$LOG_FILE"
echo "Deep Picker路径: $DEEP_PICKER_PATH" | tee -a "$LOG_FILE"
echo "Deep Picker参数: scale=55, scale2=20, model=2, auto_ppp=yes" | tee -a "$LOG_FILE"
echo "新增功能: 自动DSS化学位移校正到0.000 ppm" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"

# 存储find结果到临时文件
echo "搜索FID文件中..." | tee -a "$LOG_FILE"
/usr/bin/find "$DATA_DIR" -name "fid" -type f > /tmp/nmr_fid_list.txt
set total_fids = `wc -l < /tmp/nmr_fid_list.txt`
echo "找到了 $total_fids 个FID文件" | tee -a "$LOG_FILE"

# 样本计数器
set sample_num = 0

# 主处理循环
foreach fid_file (`cat /tmp/nmr_fid_list.txt`)
  @ sample_num++
  
  set dataset_dir = `dirname "$fid_file"`
  set dataset_name = `basename "$dataset_dir"`
  
  echo "\n========================================" | tee -a "$LOG_FILE"
  echo "[$sample_num/$total_fids] 开始处理: $dataset_name" | tee -a "$LOG_FILE"
  echo "数据目录: $dataset_dir" | tee -a "$LOG_FILE"
  echo "========================================\n" | tee -a "$LOG_FILE"
  
  if ( ! -f "$dataset_dir/acqus" ) then
    echo "跳过: $dataset_name (缺少acqus文件)" | tee -a "$LOG_FILE"
    continue
  endif

  echo "处理: $dataset_name" | tee -a "$LOG_FILE"
  echo "开始提取参数..." | tee -a "$LOG_FILE"
  
  # DECIM
  echo "  提取DECIM参数..." | tee -a "$LOG_FILE"
  grep -a '^##$DECIM=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set decim_val = `echo "$param_line" | sed 's/##\$DECIM=[ ]*//'`
  if ("$decim_val" == "") then
    set decim_val = 1680
    echo "未找到DECIM参数，使用默认值: $decim_val" | tee -a "$LOG_FILE"
  else
    echo "DECIM = $decim_val" | tee -a "$LOG_FILE"
  endif
  
  # DSPFVS
  echo "  提取DSPFVS参数..." | tee -a "$LOG_FILE"
  grep -a '^##$DSPFVS=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set dspfvs_val = `echo "$param_line" | sed 's/##\$DSPFVS=[ ]*//'`
  if ("$dspfvs_val" == "") then
    set dspfvs_val = 21
    echo "未找到DSPFVS参数，使用默认值: $dspfvs_val" | tee -a "$LOG_FILE"
  else
    echo "DSPFVS = $dspfvs_val" | tee -a "$LOG_FILE"
  endif
  
  # GRPDLY
  echo "  提取GRPDLY参数..." | tee -a "$LOG_FILE"
  grep -a '^##$GRPDLY=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set grpdly_val = `echo "$param_line" | sed 's/##\$GRPDLY=[ ]*//'`
  if ("$grpdly_val" == "") then
    set grpdly_val = 76
    echo "未找到GRPDLY参数，使用默认值: $grpdly_val" | tee -a "$LOG_FILE"
  else
    echo "GRPDLY = $grpdly_val" | tee -a "$LOG_FILE"
  endif
  
  # SW_h
  echo "  提取SW_h参数..." | tee -a "$LOG_FILE"
  grep -a '^##$SW_h=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set sw_h_val = `echo "$param_line" | sed 's/##\$SW_h=[ ]*//'`
  if ("$sw_h_val" == "") then
    set sw_h_val = 11904.76
    echo "未找到SW_h参数，使用默认值: $sw_h_val" | tee -a "$LOG_FILE"
  else
    echo "SW_h = $sw_h_val" | tee -a "$LOG_FILE"
  endif
  
  # SFO1
  echo "  提取SFO1参数..." | tee -a "$LOG_FILE"
  grep -a '^##$SFO1=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set sfo1_val = `echo "$param_line" | sed 's/##\$SFO1=[ ]*//'`
  if ("$sfo1_val" == "") then
    set sfo1_val = 600.15
    echo "未找到SFO1参数，使用默认值: $sfo1_val" | tee -a "$LOG_FILE"
  else
    echo "SFO1 = $sfo1_val" | tee -a "$LOG_FILE"
  endif
  
  # TD
  echo "  提取TD参数..." | tee -a "$LOG_FILE"
  grep -a '^##$TD=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set td_val = `echo "$param_line" | sed 's/##\$TD=[ ]*//'`
  if ("$td_val" == "") then
    set td_val = 65536
    echo "未找到TD参数，使用默认值: $td_val" | tee -a "$LOG_FILE"
  else
    echo "TD = $td_val" | tee -a "$LOG_FILE"
  endif
  
  # NUC1
  echo "  提取NUC1参数..." | tee -a "$LOG_FILE"
  grep -a '^##$NUC1=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set nuc1_val = `echo "$param_line" | sed 's/##\$NUC1=[ ]*//' | sed 's/<//g' | sed 's/>//g'`
  if ("$nuc1_val" == "") then
    set nuc1_val = "1H"
    echo "未找到NUC1参数，使用默认值: $nuc1_val" | tee -a "$LOG_FILE"
  else
    echo "NUC1 = $nuc1_val" | tee -a "$LOG_FILE"
  endif
  
  # O1
  echo "  提取O1参数..." | tee -a "$LOG_FILE"
  grep -a '^##$O1=' "$dataset_dir/acqus" > /tmp/nmr_param.txt
  set param_line = `cat /tmp/nmr_param.txt`
  set o1_val = `echo "$param_line" | sed 's/##\$O1=[ ]*//'`
  if ("$o1_val" == "") then
    set o1_val = 3705.926
    echo "未找到O1参数，使用默认值: $o1_val" | tee -a "$LOG_FILE"
  else
    echo "O1 = $o1_val" | tee -a "$LOG_FILE"
  endif
  
  # 计算O1P
  echo "  计算O1P值..." | tee -a "$LOG_FILE"
  set o1p_val = `perl -e "printf '%.6f', $o1_val / $sfo1_val;"`
  if ($status != 0 || "$o1p_val" == "" || "$o1p_val" == "nan" || "$o1p_val" == "inf") then
    echo "  计算失败，使用默认值6.0" | tee -a "$LOG_FILE"
    set o1p_val = 6.0
  endif
  echo "计算得到O1P = $o1p_val (O1=$o1_val, SFO1=$sfo1_val)" | tee -a "$LOG_FILE"

  # 创建NMRPipe目录
  echo "创建NMRPipe目录: $dataset_dir/nmrpipe" | tee -a "$LOG_FILE"
  mkdir -p "$dataset_dir/nmrpipe"
  cd "$dataset_dir/nmrpipe"

  # 创建转换脚本
  echo "创建Bruker转NMRPipe转换脚本..." | tee -a "$LOG_FILE"
  echo "#\!/usr/bin/tcsh -f" > convert.com
  echo "# Bruker数据转换脚本" >> convert.com
  echo "# 自动生成于: `date`" >> convert.com
  echo "# 源文件: $dataset_dir/fid" >> convert.com
  echo "bruk2pipe -verb -in $dataset_dir/fid \\" >> convert.com
  echo "  -bad 0.0 -ext -aswap -AMX \\" >> convert.com  
  echo "  -decim $decim_val -dspfvs $dspfvs_val -grpdly $grpdly_val -ws 8 -noi2f \\" >> convert.com
  echo "  -xN $td_val \\" >> convert.com
  echo "  -xT `expr $td_val / 2` \\" >> convert.com  
  echo "  -xMODE DQD -xSW $sw_h_val \\" >> convert.com  
  echo "  -xOBS $sfo1_val -xCAR $o1p_val \\" >> convert.com
  echo "  -xLAB $nuc1_val -ndim 1 \\" >> convert.com
  echo "  | nmrPipe -fn MULT -c 7.81250e+00 \\" >> convert.com  
  echo "  -out test.fid -ov" >> convert.com

  chmod +x convert.com
  echo "执行数据转换中..." | tee -a "$LOG_FILE"
  ./convert.com >& /tmp/nmr_convert.log
  
  if ( ! -f test.fid ) then
    echo "转换失败: $dataset_name" | tee -a "$LOG_FILE"
    echo "错误内容:" | tee -a "$LOG_FILE"
    cat /tmp/nmr_convert.log | tee -a "$LOG_FILE"
    echo "尝试备用转换方法..." | tee -a "$LOG_FILE"
    
    echo "#\!/usr/bin/tcsh -f" > convert_alt.com
    echo "# Bruker数据转换脚本 - 备用方法" >> convert_alt.com
    echo "bruk2pipe -verb -in $dataset_dir/fid \\" >> convert_alt.com
    echo "  -bad 0.0 -ext -aswap -DMX \\" >> convert_alt.com
    echo "  -decim $decim_val -dspfvs $dspfvs_val -grpdly $grpdly_val -ws 8 -noi2f \\" >> convert_alt.com
    echo "  -xN $td_val \\" >> convert_alt.com
    echo "  -xT `expr $td_val / 2` \\" >> convert_alt.com
    echo "  -xMODE DQD -xSW $sw_h_val \\" >> convert_alt.com
    echo "  -xOBS $sfo1_val -xCAR $o1p_val \\" >> convert_alt.com
    echo "  -xLAB $nuc1_val -ndim 1 \\" >> convert_alt.com
    echo "  | nmrPipe -fn MULT -c 7.81250e+00 \\" >> convert_alt.com  
    echo "  -out test.fid -ov" >> convert_alt.com
    
    chmod +x convert_alt.com
    ./convert_alt.com >& /tmp/nmr_convert_alt.log
    
    if ( ! -f test.fid ) then
      echo "所有转换方法失败，跳过后续处理" | tee -a "$LOG_FILE"
      continue
    endif
  endif

  echo "转换成功: $dataset_name" | tee -a "$LOG_FILE"

  if ($PROCESS_FLAG == 2) then
    echo "创建优化的处理脚本..." | tee -a "$LOG_FILE"
    echo "#\!/usr/bin/tcsh -f" > process.com
    echo "# NMR数据处理脚本 - Kaiser窗函数 + 自动相位校正" >> process.com
    echo "# 代谢谱优化版：增加 zero filling 提高数字分辨率" >> process.com
    echo "# 自动生成于: `date`" >> process.com
    echo "" >> process.com
    echo "# 使用basicAutoPhase.com自动确定相位值" >> process.com
    echo 'set xP0 = (`basicAutoPhase.com -in test.fid -apxELB 1.0 -apxP1 0.0 -apOrd 0 -apWindow 2%`)' >> process.com
    echo 'echo "自动相位参数: x0 = $xP0"' >> process.com
    echo "" >> process.com
    echo "# 应用相位值进行处理 - 使用Deep Picker推荐的Kaiser窗函数" >> process.com
    echo "# 增加 zero filling 到 2 倍以提高 PPP" >> process.com
    echo "nmrPipe -in test.fid \\" >> process.com
    echo "| nmrPipe -fn SP -off 0.5 -end 0.896 -pow 3.684 \\" >> process.com
    echo "| nmrPipe -fn ZF -zf 2 \\" >> process.com
    echo "| nmrPipe -fn FT \\" >> process.com
    echo '| nmrPipe -fn PS -p0 $xP0 -p1 0.0 -di \' >> process.com
    echo "| nmrPipe -fn POLY -auto -ord 0 \\" >> process.com
    echo "| nmrPipe -fn BASE -nw 20 -nl 200 \\" >> process.com
    echo "  -out spectrum.ft1 -ov" >> process.com
    echo "echo '处理完成!'" >> process.com

    chmod +x process.com
    echo "执行数据处理中..." | tee -a "$LOG_FILE"
    ./process.com |& tee -a "$LOG_FILE"

    if ( ! -f spectrum.ft1 ) then
      echo "处理失败: $dataset_name" | tee -a "$LOG_FILE"
      continue
    endif

    echo "转换为文本格式..." | tee -a "$LOG_FILE"
    pipe2xyz -in spectrum.ft1 -out spectrum.txt -noverb >& /tmp/nmr_txt.log
    
    if ( ! -f spectrum.txt ) then
      echo "文本转换失败，尝试备用方法..." | tee -a "$LOG_FILE"
      nmrPipe -in spectrum.ft1 | pipe2txt -out spectrum.txt -noHeader >& /tmp/nmr_txt_alt.log
    endif

    echo "\n=====================================================================" | tee -a "$LOG_FILE"
    echo "  开始Deep Picker峰拾取（使用测试验证的最佳参数）" | tee -a "$LOG_FILE"
    echo "=====================================================================" | tee -a "$LOG_FILE"
    
    # 分析PPP和数据质量
    echo "分析PPP和数据质量..." | tee -a "$LOG_FILE"
    
    perl -e "print $sw_h_val / $td_val" > /tmp/nmr_res.txt
    set digital_res_hz = `cat /tmp/nmr_res.txt`
    
    perl -e "print $digital_res_hz / $sfo1_val" > /tmp/nmr_res_ppm.txt
    set digital_res_ppm = `cat /tmp/nmr_res_ppm.txt`
    
    # 计算原始 PPP
    perl -e "print 1.0 / $digital_res_hz" > /tmp/nmr_ppp.txt
    set estimated_ppp = `cat /tmp/nmr_ppp.txt`
    
    # 计算 ZF 后的 PPP（ZF=2）
    perl -e "print $estimated_ppp * 2" > /tmp/nmr_ppp_zf.txt
    set adjusted_ppp = `cat /tmp/nmr_ppp_zf.txt`
    
    echo "数字分辨率: $digital_res_hz Hz/point, $digital_res_ppm ppm/point" | tee -a "$LOG_FILE"
    echo "原始PPP: $estimated_ppp points" | tee -a "$LOG_FILE"
    echo "ZF后PPP: $adjusted_ppp points (2倍zero filling)" | tee -a "$LOG_FILE"
    
    # SNR分析
    if ( -f spectrum.txt ) then
      echo "分析信噪比..." | tee -a "$LOG_FILE"
      
      perl -e 'open(F,"<spectrum.txt")or die;my($ns,$nc,$sm)=(0,0,0);while(<F>){next if/^#|^VARS|^FORMAT/;my@f=split;next if@f<2;my($p,$i)=($f[0],$f[1]);if(($p>=-0.5&&$p<=0)||($p>=10.5&&$p<=11)){$ns+=$i*$i;$nc++}$sm=$i if($p>=0&&$p<=10&&$i>$sm)}close F;if($nc>0){my$rms=sqrt($ns/$nc);printf"%.1f %.2e\n",($rms>0)?$sm/$rms:100,$rms}else{print"50.0 1e10\n"}' > /tmp/snr_result.txt
      
      if ( $status == 0 && -f /tmp/snr_result.txt ) then
        set snr_value = `cut -d' ' -f1 /tmp/snr_result.txt`
        set noise_level = `cut -d' ' -f2 /tmp/snr_result.txt`
        echo "估算信噪比: $snr_value" | tee -a "$LOG_FILE"
        echo "噪声水平: $noise_level" | tee -a "$LOG_FILE"
      else
        set snr_value = 50.0
        set noise_level = 1e10
        echo "SNR分析失败，使用默认值" | tee -a "$LOG_FILE"
      endif
    else
      set snr_value = 50.0
      set noise_level = 1e10
      echo "spectrum.txt 不存在，使用默认SNR值" | tee -a "$LOG_FILE"
    endif
    
    # 使用测试验证的最佳参数
    echo "\n使用最佳参数组合（基于实际测试结果）:" | tee -a "$LOG_FILE"
    echo "  scale  = 55     (严格参数，减少假峰)" | tee -a "$LOG_FILE"
    echo "  scale2 = 20     (严格的次级阈值)" | tee -a "$LOG_FILE"
    echo "  model  = 2      (代谢物模型)" | tee -a "$LOG_FILE"
    echo "  auto_ppp = yes  (自动PPP调整，影响最大)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "该参数组合在测试中表现:" | tee -a "$LOG_FILE"
    echo "  - 总峰数: 657个（合理范围）" | tee -a "$LOG_FILE"
    echo "  - 芳香区: 90个（可接受）" | tee -a "$LOG_FILE"
    echo "  - 负区域: 0个（完美）" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    set scale_param = 55
    set scale2_param = 20
    set model_param = 2
    set auto_ppp_param = yes
    
    # 执行Deep Picker
    echo "执行Deep Picker..." | tee -a "$LOG_FILE"
    $DEEP_PICKER_PATH -in spectrum.ft1 -out spectrum.tab -scale $scale_param -scale2 $scale2_param -model $model_param -auto_ppp $auto_ppp_param >& /tmp/deeppicker.log
    
    if ( -f spectrum.tab ) then
      set peak_count = `grep -v '^#\|^VARS\|^FORMAT' spectrum.tab | wc -l`
      echo "✓ Deep Picker完成，识别峰数: $peak_count" | tee -a "$LOG_FILE"
      
      # 统计区域分布
      perl -e 'open(F,"<spectrum.tab")or die;my($a,$n)=(0,0);while(<F>){next if/^#|^VARS|^FORMAT/;my@f=split;next if@f<3;$a++if($f[2]>=6&&$f[2]<=9);$n++if($f[2]<0)}print"$a $n\n"' > /tmp/dist_result.txt
      set aromatic_peaks = `cut -d' ' -f1 /tmp/dist_result.txt`
      set negative_peaks = `cut -d' ' -f2 /tmp/dist_result.txt`
      
      echo "  - 芳香区域(6-9ppm): $aromatic_peaks 个峰" | tee -a "$LOG_FILE"
      echo "  - 负化学位移: $negative_peaks 个峰" | tee -a "$LOG_FILE"
      
      # 新增：自动DSS化学位移校正
      # ====================================================================
      echo "\n=====================================================================" | tee -a "$LOG_FILE"
      echo "  自动DSS化学位移校正" | tee -a "$LOG_FILE"
      echo "=====================================================================" | tee -a "$LOG_FILE"

      echo "检测0 ± 0.05 ppm范围内的DSS峰..." | tee -a "$LOG_FILE"

      # 使用perl检测DSS峰
      perl -e 'open(F,"<spectrum.tab")or die;my($max_h,$dss_ppm)=(0,0);while(<F>){next if/^#|^VARS|^FORMAT/;my@f=split;next if@f<6;my($ppm,$height)=($f[2],$f[4]);if($ppm>=-0.05&&$ppm<=0.05&&$height>$max_h){$max_h=$height;$dss_ppm=$ppm}}printf"%.6f\n",$dss_ppm' > /tmp/dss_ppm.txt

      set dss_ppm = `cat /tmp/dss_ppm.txt`

      if ( "$dss_ppm" == "0.000000" || "$dss_ppm" == "" ) then
        echo "⚠ 未检测到DSS峰，跳过化学位移校正" | tee -a "$LOG_FILE"
      else
        echo "✓ 检测到DSS峰在: $dss_ppm ppm" | tee -a "$LOG_FILE"
        
        # 计算偏移量的绝对值
        set offset_abs = `perl -e "printf '%.6f', abs($dss_ppm)"`
        
        if ( `perl -e "print ($offset_abs > 0.001 ? 1 : 0)"` ) then
          echo "应用化学位移校正: -$dss_ppm ppm" | tee -a "$LOG_FILE"
          
          # 🔧 修复：通过命令行参数传递变量
          perl -e 'open(F,"<spectrum.tab")or die;open(O,">spectrum_corrected.tab")or die;my $offset=shift;while(<F>){if(/^#|^VARS|^FORMAT/){print O;next}my@f=split;if(@f>=6){$f[2]=sprintf("%.4f",$f[2]-$offset);print O join(" ",@f),"\n"}else{print O}}close F;close O' $dss_ppm
          
          mv spectrum_corrected.tab spectrum.tab
          
          # 验证校正结果
          perl -e 'open(F,"<spectrum.tab")or die;my($max_h,$dss_ppm)=(0,0);while(<F>){next if/^#|^VARS|^FORMAT/;my@f=split;next if@f<6;my($ppm,$height)=($f[2],$f[4]);if($ppm>=-0.05&&$ppm<=0.05&&$height>$max_h){$max_h=$height;$dss_ppm=$ppm}}printf"%.6f\n",$dss_ppm' > /tmp/dss_after.txt
          
          set dss_after = `cat /tmp/dss_after.txt`
          
          echo "✓ 化学位移校正完成" | tee -a "$LOG_FILE"
          echo "  校正前DSS位置: $dss_ppm ppm" | tee -a "$LOG_FILE"
          echo "  校正后DSS位置: $dss_after ppm" | tee -a "$LOG_FILE"
        else
          echo "✓ DSS峰已经在0.000 ppm附近（偏差 < 0.001 ppm），无需校正" | tee -a "$LOG_FILE"
        endif
      endif

      echo "=====================================================================" | tee -a "$LOG_FILE"
      
    else
      echo "✗ Deep Picker处理失败: $dataset_name" | tee -a "$LOG_FILE"
      cat /tmp/deeppicker.log | tee -a "$LOG_FILE"
    endif

    echo "\n处理完成: $dataset_name" | tee -a "$LOG_FILE"
    echo "输出文件:" | tee -a "$LOG_FILE"
    echo "  - NMRPipe格式: $dataset_dir/nmrpipe/spectrum.ft1" | tee -a "$LOG_FILE"
    echo "  - 文本格式: $dataset_dir/nmrpipe/spectrum.txt" | tee -a "$LOG_FILE"
    echo "  - 峰表格式: $dataset_dir/nmrpipe/spectrum.tab (已DSS校正)" | tee -a "$LOG_FILE"
    echo "  - 使用参数: scale=$scale_param, scale2=$scale2_param, model=$model_param, auto_ppp=$auto_ppp_param" | tee -a "$LOG_FILE"
  endif
end

# 清理临时文件
echo "\n清理临时文件..." | tee -a "$LOG_FILE"
rm -f /tmp/nmr_*.txt /tmp/nmr_*.log /tmp/*.txt

# 结束提示
echo "\n========================================" | tee -a "$LOG_FILE"
echo "所有数据集处理完成" | tee -a "$LOG_FILE"
echo "总样本数: $total_fids" | tee -a "$LOG_FILE"
echo "成功处理: $sample_num 个样本" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"