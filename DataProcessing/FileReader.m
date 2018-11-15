%RowOffset Should be 10 (Passes the information columns
%Threshold is cutoff for a viable signal
% Tansmitter is current data set

function FileReader(rowOffset, threshold, transmitter)
    F = dir('*.csv');
    for curr = 1:length(F)
        DataArray = csvread(F(curr).name, rowOffset, 0);
        rowSize = size(DataArray);
        IVector = ones(rowSize(1),1);
        QVector = ones(rowSize(1),1);
        for i = 1:rowSize(1)
            if (DataArray(i,1) >= threshold && DataArray(i,2) >= threshold)
                IVector(i) = DataArray(i,1);
                QVector(i) = DataArray(i,2);
                %fprintf('%d %d\n', DataArray(i,1), DataArray(i,2));
            else
                IVector(i) = -99999;
                QVector(i) = -99999;
            end
        end
        %getSize
        ICount = 0;
        QCount = 0;
        for i = 1:rowSize(1)
            if (IVector(i) > -99998)
                ICount = ICount + 1;
            end
            if (QVector(i) > -99998)
                QCount = QCount + 1;
            end
        end
        IData = ones(ICount, 1);
        QData = ones(QCount, 1);
        ICounter = 1;
        iter = 1;
        while (ICounter <= ICount)
            if (IVector(iter) > -99998)
                IData(ICounter) = IVector(iter);
                ICounter = ICounter + 1;
            end
            iter = iter+1;
        end
        iter = 1;
        QCounter = 1;
        while (QCounter <= QCount)
            if (QVector(iter) > -99998)
                QData(QCounter) = QVector(iter);
                QCounter = QCounter + 1;
            end
            iter = iter+1;
        end
        toWrite = [IData QData];
        filename = strcat('Output\',transmitter, '_', int2str(curr), '.csv');
        dlmwrite(filename, toWrite);
    end
end
    
    