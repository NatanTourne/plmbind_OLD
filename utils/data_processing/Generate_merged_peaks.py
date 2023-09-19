from numpy import isin


delta_list = []
uniques_list = []
count = 0
with open("/home/data/shared/natant/Data/remap2022_nr_macs2_hg38_v1_0.bed") as f:
    for line in f:
        count += 1
        if count > 100000000000000000:
            break
        
        line_list = line.split()
        TF = line_list[3].split(':')[0]
        start = int(line_list[1])
        stop = int(line_list[2])
        pos = line_list[0:3]
        chrom = line_list[0]
        
        if count == 1:
            region_start = start
            region_stop = stop
            region_TFs = [TF]
            region_chrom = chrom
        else:
            if start > region_stop or chrom != region_chrom:
                delta_list.append(region_stop-region_start)
                uniques_list.append(len(region_TFs))

                with open("/home/data/shared/natant/Data/remap_merged_peaks.bed", 'a') as out_file:
                    out_file.write(region_chrom + "\t" + str(region_start) + "\t" + str(region_stop) + "\t" + ",".join(region_TFs) + "\n")
                    
                region_start = start
                region_stop = stop
                region_TFs = [TF]
                region_chrom = chrom
                
                
            else:
                region_stop = stop
                if not TF in region_TFs:
                    region_TFs.append(TF)