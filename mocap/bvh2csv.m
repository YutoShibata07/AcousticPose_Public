function bvh2csv(bvh_name, csv_name)


    [skeleton, ~] = loadbvh(bvh_name);
    frame_num = size(skeleton(1).Dxyz, 2);
    keypoint_num = size(skeleton, 2);
    
    
    %% generate data matrix
    pos_mat = zeros(frame_num, 3*keypoint_num);
    for ID = 1 : keypoint_num
        pos_tmp = skeleton(ID).Dxyz;
        pos_mat(:, 3*(ID-1)+1:3*ID) = pos_tmp';
    end

    
    %% export csv 
    fileID = fopen(csv_name,'w');
    
    % write key point name
    for ID = 1 : keypoint_num
        fprintf(fileID, '%s, %s, %s,', skeleton(ID).name, skeleton(ID).name, skeleton(ID).name);
    end
    fprintf(fileID, '\n');
    
    % write axis
    for ID = 1 : keypoint_num
        fprintf(fileID, 'x, y, z,');
    end
    fprintf(fileID, '\n');
    
    % write position data
    for j = 1 : frame_num
        for i = 1 : keypoint_num
            fprintf(fileID, '%f, %f, %f,', pos_mat(j, 3*(i-1)+1), pos_mat(j, 3*(i-1)+2), pos_mat(j, 3*(i-1)+3));
        end
        fprintf(fileID, '\n');
    end
    
    fclose(fileID);

end